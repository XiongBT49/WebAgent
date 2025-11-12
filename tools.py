"""
Pure Python Toolchain for VLLM Web Agent
Implements all required tools without external runtime dependencies like MCP
"""

import os
import time
from typing import Any, Dict, Optional
from pathlib import Path
import io
import requests

from playwright.sync_api import sync_playwright, Page, Browser, TimeoutError
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import PyPDF2
from PIL import Image

import config


class BrowserTools:
    """Browser control tools using Playwright"""

    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None
        self.context = None

    def initialize(self):
        """Initialize browser instance"""
        if self.playwright is None:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(
                headless=config.BROWSER_HEADLESS
            )
            self.context = self.browser.new_context(
                viewport={"width": 1280, "height": 720}
            )
            self.page = self.context.new_page()
            self.page.set_default_timeout(config.BROWSER_TIMEOUT)

    def cleanup(self):
        """Clean up browser resources"""
        if self.page:
            self.page.close()
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def goto(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL"""
        try:
            self.initialize()
            response = self.page.goto(url, wait_until="domcontentloaded")
            time.sleep(1)  # Wait for page to settle
            return {
                "success": True,
                "url": self.page.url,
                "title": self.page.title(),
                "status": response.status if response else None,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def click(self, selector: str) -> Dict[str, Any]:
        """
        Click an element. Tries the CSS selector first, then falls back
        to finding an element by its visible text.
        **This action does NOT wait for navigation. The agent is responsible
        for waiting and checking the DOM after the click.**
        """
        if self.page is None:
            return {
                "success": False,
                "error": "Browser page not initialized. Use 'goto' first.",
            }

        locator_to_click = None

        # --- Step 1: Find the element to click ---
        try:
            locator_to_click = self.page.locator(selector)
            locator_to_click.wait_for(state="visible", timeout=5000)
            print(f"Info: Found element using selector '{selector}'.")
        except Exception:
            print(
                f"Info: Could not find with selector '{selector}'. Trying to find by text..."
            )
            try:
                locator_to_click = self.page.get_by_text(selector).first
                locator_to_click.wait_for(state="visible", timeout=5000)
                print(f"Info: Found element with text '{selector}'.")
            except Exception as e:
                error_message = f"Failed to find element using selector '{selector}' and by text. Error: {str(e)}"
                print(f"Error: {error_message}")
                return {"success": False, "error": error_message, "selector": selector}

        # --- Step 2: Perform the click without waiting for navigation ---
        try:
            locator_to_click.click(no_wait_after=True)
            # Action is considered successful once the click is dispatched.
            # A small, fixed wait can help prevent race conditions in the agent.
            time.sleep(1)
            return {"success": True, "message": "Click action dispatched. Agent should wait and check DOM."}
        except Exception as e:
            error_message = f"An unexpected error occurred during click. Error: {str(e)}"
            print(f"Error: {error_message}")
            return {"success": False, "error": error_message}

    def download_pdf(self, pdf_url: Optional[str] = None, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Downloads a PDF file using Playwright's download handling.
        ALWAYS uses Playwright to click download buttons - no direct HTTP download!
        
        Workflow:
        1. If pdf_url provided: navigate to it first (usually an abstract page)
        2. Find and click the download button/link on the page
        3. Capture the download event
        4. Save the downloaded file
        
        Args:
            pdf_url: URL to navigate to (abstract page with download button)
            filename: Desired filename for the downloaded PDF (optional)
        """
        if self.page is None:
            return {
                "success": False,
                "error": "Browser page not initialized. Use 'goto' first.",
            }

        try:
            # Ensure output directory exists
            os.makedirs(config.PDFS_DIR, exist_ok=True)
            
            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"downloaded_{timestamp}.pdf"
            
            # Ensure .pdf extension
            if not filename.lower().endswith('.pdf'):
                filename += '.pdf'
            
            filepath = os.path.join(config.PDFS_DIR, filename)
            
            # Step 1: Navigate to the page if pdf_url provided
            if pdf_url:
                print(f"Info: Navigating to: {pdf_url}")
                # For arXiv, if it's a direct PDF link, convert to abstract page
                if '/pdf/' in pdf_url:
                    # Convert https://arxiv.org/pdf/2409.12191 to https://arxiv.org/abs/2409.12191
                    abstract_url = pdf_url.replace('/pdf/', '/abs/')
                    if abstract_url.endswith('.pdf'):
                        abstract_url = abstract_url[:-4]
                    print(f"Info: Converted PDF URL to abstract page: {abstract_url}")
                    self.page.goto(abstract_url, wait_until="domcontentloaded")
                else:
                    self.page.goto(pdf_url, wait_until="domcontentloaded")
                time.sleep(2)  # Wait for page to load
            
            # Step 2: Start waiting for download and click the download button
            print(f"Info: Looking for download button on: {self.page.url}")
            
            with self.page.expect_download(timeout=30000) as download_info:
                # Try multiple selectors for arXiv download buttons
                selectors_to_try = [
                    "a.download-pdf",
                    "a[href*='/pdf/']",
                    "a:has-text('Download PDF')",
                    "a:has-text('PDF')",
                    ".full-text a",
                    "a[accesskey='f']",  # arXiv PDF download link
                ]
                
                clicked = False
                for selector in selectors_to_try:
                    try:
                        locator = self.page.locator(selector).first
                        locator.wait_for(state="visible", timeout=3000)
                        
                        # Get the href to verify it's a PDF link
                        href = locator.get_attribute('href')
                        if href and '/pdf/' in href:
                            print(f"Info: Found PDF download link using selector: {selector}")
                            print(f"Info: Link href: {href}")
                            locator.click()
                            clicked = True
                            break
                    except Exception as e:
                        continue
                
                if not clicked:
                    return {
                        "success": False,
                        "error": f"Could not find download button on page: {self.page.url}. Try using 'screenshot' to see the page visually.",
                    }
            
            # Step 3: Save the downloaded file
            download = download_info.value
            download.save_as(filepath)
            
            print(f"Info: Successfully downloaded PDF to: {filepath}")
            return {
                "success": True,
                "filepath": filepath,
                "filename": filename,
            }
            
        except TimeoutError:
            return {
                "success": False,
                "error": f"Download timeout after clicking. The download may not have started. Page: {self.page.url}",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to download PDF: {str(e)}"}

    def close(self) -> None:
        """
        Close the browser.
        This method is called automatically when the BrowserTools instance is deleted.
        """
        try:
            if self.page:
                self.page.close()
                print("Info: Closed the current page.")
            if self.context:
                self.context.close()
                print("Info: Closed the browser context.")
            if self.browser:
                self.browser.close()
                print("Info: Closed the browser.")
            if self.playwright:
                self.playwright.stop()
                print("Info: Stopped Playwright.")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")


    def type_text(
        self, selector: str, text: str, press_enter: bool = False
    ) -> Dict[str, Any]:
        """
        Type text into an element and optionally press Enter.
        **This action does NOT wait for navigation. The agent is responsible
        for waiting and checking the DOM after the action.**
        """
        if self.page is None:
            return {
                "success": False,
                "error": "Browser page not initialized. Use 'goto' first.",
            }

        target_locator = None
        
        # --- Find the element to type in ---
        try:
            target_locator = self.page.locator(selector)
            target_locator.wait_for(state="visible", timeout=3000)
            print(f"Info: Found element using selector '{selector}'.")
        except Exception:
            print(
                f"Info: Primary selector '{selector}' failed. Trying common fallbacks for input fields..."
            )
            fallback_selectors = [
                'input[name="q"]', 'textarea[name="q"]', 'input[type="search"]',
                'input[type="text"][aria-label*="search" i]',
                'input[type="text"][title*="search" i]',
                'input[type="text"][placeholder*="search" i]',
            ]
            for fb_selector in fallback_selectors:
                try:
                    locator = self.page.locator(fb_selector).first
                    locator.wait_for(state="visible", timeout=100)
                    target_locator = locator
                    print(f"  -> Found element with fallback: '{fb_selector}'")
                    break
                except Exception:
                    continue
        
        if not target_locator:
            error_message = f"Failed to find a typeable element using primary selector '{selector}' and all fallbacks."
            print(f"Error: {error_message}")
            return {"success": False, "error": error_message}

        # --- Perform the typing and pressing Enter ---
        try:
            target_locator.fill(text)
            if press_enter:
                # Use no_wait_after=True to prevent the tool from hanging
                target_locator.press("Enter", no_wait_after=True)
            
            # Action is successful once keys are sent.
            time.sleep(1) # A small, fixed wait
            return {"success": True, "message": "Type action dispatched. Agent should wait and check DOM."}
        
        except Exception as e:
            error_message = f"Found element, but failed to type or press Enter. Error: {str(e)}"
            print(f"Error: {error_message}")
            return {"success": False, "error": error_message}

    def press(self, key: str) -> Dict[str, Any]:
        """Press a keyboard key"""
        try:
            self.page.keyboard.press(key)
            time.sleep(0.3)
            return {"success": True, "key": key}
        except Exception as e:
            return {"success": False, "error": str(e), "key": key}

    def wait(self, milliseconds: int) -> Dict[str, Any]:
        """Wait for specified milliseconds"""
        try:
            time.sleep(milliseconds / 1000.0)
            return {"success": True, "waited_ms": milliseconds}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def screenshot(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """Take a screenshot of the current page"""
        if self.page is None:
            return {
                "success": False,
                "error": "Browser page not initialized. Use 'goto' first.",
            }
        try:
            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"

            filepath = os.path.join(config.SCREENSHOTS_DIR, filename)
            os.makedirs(config.SCREENSHOTS_DIR, exist_ok=True)

            self.page.screenshot(path=filepath, full_page=False)
            return {"success": True, "filepath": filepath, "filename": filename}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def dom_summary(self) -> Dict[str, Any]:
        """
        Get a structured, actionable summary of the current page's DOM.
        This version is a hybrid: it uses heuristics for known sites (arXiv)
        and a general-purpose extractor for others.
        """
        if self.page is None:
            return {
                "success": False,
                "error": "Browser page not initialized. Use 'goto' first.",
            }
        try:
            # 关键修复：在获取内容前，等待页面加载稳定。
            # 这可以防止在 'click' 或 'type' 等触发导航的动作后出现竞争条件。
            self.page.wait_for_load_state("domcontentloaded", timeout=7000)
            content = self.page.content()
            soup = BeautifulSoup(content, "html.parser")
            url = self.page.url

            summary = {
                "url": url,
                "title": self.page.title(),
                "links": [],
                "buttons": [],
                "inputs": [],
                "specialized_content": {}, # 用于特定网站的结构化数据
                "text_summary": "",
            }

            # --- 基于启发式规则提取 arXiv 网站信息 ---
            if "arxiv.org" in url:
                # 摘要页面的启发式规则 (例如 /abs/...)
                if "/abs/" in url:
                    download_section = soup.find("div", class_="full-text")
                    if download_section:
                        pdf_link = download_section.find("a", class_="download-pdf")
                        if pdf_link and pdf_link.has_attr('href'):
                            summary["specialized_content"]["pdf_link"] = {
                                "text": pdf_link.get_text(strip=True),
                                "href": urljoin(url, pdf_link["href"])
                            }

                # 针对搜索结果页面的最终、健壮的启发式规则
                if "/search/" in url:
                    results_container = soup.find("ol", class_="breathe-horizontal")
                    if results_container:
                        results = []
                        for li in results_container.find_all("li", class_="arxiv-result", limit=5):
                            title_elem = li.find("p", class_="title")
                            authors_elem = li.find("p", class_="authors")
                            abs_link_elem = li.select_one("p.list-title > a")
                            pdf_link_elem = li.find("a", string="pdf")

                            result_item = {
                                "title": title_elem.get_text(strip=True) if title_elem else "N/A",
                                "authors": authors_elem.get_text(strip=True).replace("Authors:", "").strip() if authors_elem else "N/A",
                                "pdf_href": urljoin(url, pdf_link_elem["href"]) if pdf_link_elem and pdf_link_elem.has_attr('href') else None,
                                "abstract_href": urljoin(url, abs_link_elem["href"]) if abs_link_elem and abs_link_elem.has_attr('href') else None,
                            }
                            results.append(result_item)
                        
                        if results:
                            summary["specialized_content"]["search_results"] = results

            # --- 通用元素提取（对所有网站都运行） ---
            
            for element in soup(["script", "style", "meta", "noscript", "svg", "path"]):
                element.decompose()

            for a in soup.find_all("a", href=True, limit=20):
                summary["links"].append({
                    "text": a.get_text(strip=True)[:150],
                    "href": urljoin(url, a.get("href", ""))
                })

            for button in soup.find_all("button", limit=15):
                summary["buttons"].append({
                    "text": button.get_text(strip=True)[:100] or button.get("value", "") or button.get("aria-label", ""),
                    "id": button.get("id", ""),
                    "class": " ".join(button.get("class", [])),
                })

            for inp in soup.find_all(["input", "textarea", "select"], limit=15):
                input_type = inp.get("type", inp.name)
                if input_type not in ["hidden", "submit", "button", "reset"]:
                     summary["inputs"].append({
                        "type": input_type,
                        "name": inp.get("name", ""),
                        "id": inp.get("id", ""),
                        "placeholder": inp.get("placeholder", ""),
                        "aria-label": inp.get("aria-label", ""),
                    })
            
            body_text = soup.get_text(separator=" ", strip=True)
            cleaned_text = " ".join(body_text.split())
            summary["text_summary"] = cleaned_text[:1500]

            return {"success": True, "summary": summary}
        except Exception as e:
            error_message = f"Error in dom_summary for URL {self.page.url}: {str(e)}"
            print(f"ERROR: {error_message}")
            return {"success": False, "error": error_message}


class FileTools:
    """File operation tools"""

    @staticmethod
    def pdf_extract_text(pdf_path: str) -> Dict[str, Any]:
        """Extract text from a PDF file"""
        try:
            text_content = []

            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)

                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    text_content.append({"page": page_num + 1, "text": text})

            # Save extracted text
            txt_filename = Path(pdf_path).stem + "_extracted.txt"
            txt_filepath = os.path.join(config.OUTPUT_DIR, txt_filename)

            with open(txt_filepath, "w", encoding="utf-8") as f:
                for page_data in text_content:
                    f.write(f"=== Page {page_data['page']} ===\n")
                    f.write(page_data["text"])
                    f.write("\n\n")

            return {
                "success": True,
                "num_pages": num_pages,
                "text_filepath": txt_filepath,
                "pages": text_content[:3],  # Return first 3 pages as preview
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def pdf_extract_images(pdf_path: str) -> Dict[str, Any]:
        """
        Extract images/figures from a PDF file using prioritized approach:
        Priority 1: Embedded images (PyPDF2) - actual figure images
        Priority 2: Render pages only if no embedded images found
        Priority 3: Intelligent cropping to isolate Figure 1 from full page
        """
        try:
            extracted_images = []
            embedded_images = []
            os.makedirs(config.IMAGES_DIR, exist_ok=True)
            
            # Method 1: PRIORITY - Extract embedded images using PyPDF2 (real figures)
            print("Extracting embedded images using PyPDF2...")
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num in range(min(10, len(pdf_reader.pages))):
                    page = pdf_reader.pages[page_num]

                    if "/Resources" not in page or "/XObject" not in page["/Resources"]:
                        continue
                    
                    x_objects = page["/Resources"]["/XObject"].get_object()

                    for obj_name in x_objects:
                        obj = x_objects[obj_name]

                        if obj.get("/Subtype") != "/Image":
                            continue
                            
                        try:
                            width = obj.get("/Width", 0)
                            height = obj.get("/Height", 0)
                            
                            # Skip tiny images (likely icons)
                            if width < 100 or height < 100:
                                continue
                            
                            size = (width, height)
                            data = obj.get_data()

                            img_filename = f"{Path(pdf_path).stem}_page{page_num + 1}_embedded_{obj_name[1:]}.png"
                            img_filepath = os.path.join(config.IMAGES_DIR, img_filename)

                            color_space = obj.get("/ColorSpace", "")
                            
                            if color_space == "/DeviceRGB":
                                img = Image.frombytes("RGB", size, data)
                            elif color_space == "/DeviceGray":
                                img = Image.frombytes("L", size, data)
                            else:
                                try:
                                    img = Image.frombytes("RGB", size, data)
                                except:
                                    img = Image.frombytes("L", size, data)

                            img.save(img_filepath, "PNG")
                            img_info = {
                                "page": page_num + 1,
                                "filepath": img_filepath,
                                "filename": img_filename,
                                "size": size,
                                "type": "embedded_image",
                                "note": f"High-quality embedded figure from page {page_num + 1}"
                            }
                            embedded_images.append(img_info)
                            extracted_images.append(img_info)
                            print(f"✓ Extracted embedded image from page {page_num + 1}: {img_filename} ({width}x{height})")
                        except Exception as img_error:
                            continue
            
            # Method 2: FALLBACK - Render pages only if no embedded images found
            if not embedded_images:
                print("⚠ No embedded images found. Rendering full pages as fallback...")
                try:
                    from pdf2image import convert_from_path
                    
                    images = convert_from_path(pdf_path, first_page=1, last_page=3, dpi=200, fmt='png')
                    
                    for page_num, img in enumerate(images, start=1):
                        img_filename = f"{Path(pdf_path).stem}_page{page_num}_full.png"
                        img_filepath = os.path.join(config.IMAGES_DIR, img_filename)
                        
                        img.save(img_filepath, 'PNG', optimize=True, quality=85)
                        
                        extracted_images.append({
                            "page": page_num,
                            "filepath": img_filepath,
                            "filename": img_filename,
                            "size": img.size,
                            "type": "rendered_full_page",
                            "note": f"⚠ FULL PAGE {page_num} (not isolated figure) - Vision model needs to identify Figure 1 region"
                        })
                        print(f"✓ Rendered full page {page_num}: {img_filename}")
                        
                except Exception as pdf2image_error:
                    print(f"Error: pdf2image failed: {pdf2image_error}")
                    return {
                        "success": False,
                        "error": f"Failed to extract images: {pdf2image_error}"
                    }
            else:
                print(f"✓ Successfully extracted {len(embedded_images)} embedded images (preferred method)")

            # Prepare result with priority indication
            result = {
                "success": True,
                "num_images": len(extracted_images),
                "images": extracted_images,
                "extraction_method": "embedded_images" if embedded_images else "rendered_pages",
            }
            
            if embedded_images:
                result["note"] = f"✓ Extracted {len(embedded_images)} high-quality embedded figures. These are isolated figures, not full pages."
            else:
                result["note"] = "⚠ No embedded images found. Extracted full page renders instead. Vision model must identify Figure 1 within the page."
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def save_image(image_url: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """Save an image from a URL"""
        try:
            import requests

            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                ext = image_url.split(".")[-1].split("?")[0]
                if ext not in ["jpg", "jpeg", "png", "gif", "webp"]:
                    ext = "png"
                filename = f"image_{timestamp}.{ext}"

            filepath = os.path.join(config.IMAGES_DIR, filename)
            os.makedirs(config.IMAGES_DIR, exist_ok=True)

            response = requests.get(image_url, timeout=30)
            response.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(response.content)

            # Get image info
            img = Image.open(io.BytesIO(response.content))

            return {
                "success": True,
                "filepath": filepath,
                "filename": filename,
                "size": img.size,
                "format": img.format,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def write_text(content: str, filename: str) -> Dict[str, Any]:
        """Write text content to a file"""
        try:
            filepath = os.path.join(config.OUTPUT_DIR, filename)
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            return {
                "success": True,
                "filepath": filepath,
                "filename": filename,
                "size_bytes": len(content.encode("utf-8")),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def analyze_image(image_path: str, question: str = "Describe this image in detail.") -> Dict[str, Any]:
        """
        Analyze an image using vision model (placeholder - actual analysis done in main.py)
        This tool marks that image analysis is needed.
        """
        try:
            # Verify image exists
            if not os.path.exists(image_path):
                return {"success": False, "error": f"Image not found: {image_path}"}
            
            # Return a flag that main.py will handle
            return {
                "success": True,
                "needs_vision_analysis": True,
                "image_path": image_path,
                "question": question,
                "message": "Vision analysis will be performed..."
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# Tool registry for easy access
TOOLS = {
    # Browser control
    "goto": {
        "function": lambda browser, **kwargs: browser.goto(**kwargs),
        "description": "Navigate to a URL",
        "parameters": {"url": "The URL to navigate to"},
    },
    "click": {
        "function": lambda browser, **kwargs: browser.click(**kwargs),
        "description": "Click an element by CSS selector",
        "parameters": {"selector": "CSS selector of the element to click"},
    },
    "type": {
        "function": lambda browser, **kwargs: browser.type_text(**kwargs),
        "description": "Type text into an input field",
        "parameters": {
            "selector": "CSS selector of the input element",
            "text": "Text to type",
        },
    },
    "press": {
        "function": lambda browser, **kwargs: browser.press(**kwargs),
        "description": "Press a keyboard key",
        "parameters": {"key": "Key to press (e.g., 'Enter', 'Escape', 'ArrowDown')"},
    },
    "wait": {
        "function": lambda browser, **kwargs: browser.wait(**kwargs),
        "description": "Wait for specified milliseconds",
        "parameters": {"milliseconds": "Time to wait in milliseconds"},
    },
    "screenshot": {
        "function": lambda browser, **kwargs: browser.screenshot(**kwargs),
        "description": "Take a screenshot of the current page",
        "parameters": {"filename": "(Optional) Filename for the screenshot"},
    },
    "dom_summary": {
        "function": lambda browser, **kwargs: browser.dom_summary(**kwargs),
        "description": "Get a simplified summary of the page DOM",
        "parameters": {},
    },
    "download_pdf": {
        "function": lambda browser, **kwargs: browser.download_pdf(**kwargs),
        "description": "Downloads a PDF by navigating to the page and clicking the download button using Playwright. Provide pdf_url to navigate first (e.g., abstract page URL), then it will find and click the download button.",
        "parameters": {
            "pdf_url": "string (optional, URL to navigate to - will be converted from PDF URL to abstract page if needed)",
            "filename": "string (optional, e.g., 'my_report.pdf')"
        },
    },
    # File operations
    "pdf_extract_text": {
        "function": lambda browser, **kwargs: FileTools.pdf_extract_text(**kwargs),
        "description": "Extract text from a downloaded PDF file",
        "parameters": {"pdf_path": "Path to the PDF file (returned by 'download_pdf')"},
    },
    "pdf_extract_images": {
        "function": lambda browser, **kwargs: FileTools.pdf_extract_images(**kwargs),
        "description": "Extract images from a downloaded PDF file",
        "parameters": {"pdf_path": "Path to the PDF file (returned by 'download_pdf')"},
    },
    "save_image": {
        "function": lambda browser, **kwargs: FileTools.save_image(**kwargs),
        "description": "Save an image from a URL",
        "parameters": {
            "image_url": "URL of the image to save",
            "filename": "(Optional) Filename to save as",
        },
    },
    "write_text": {
        "function": lambda browser, **kwargs: FileTools.write_text(**kwargs),
        "description": "Saves a string of text to a file in the output directory.",
        "parameters": {"content": "string", "filename": "string"},
    },
    "analyze_image": {
        "function": lambda browser, **kwargs: FileTools.analyze_image(**kwargs),
        "description": "Analyze an image (e.g., Figure 1 extracted from PDF) using vision model. Describes visual content, charts, diagrams.",
        "parameters": {
            "image_path": "Path to the image file to analyze",
            "question": "(Optional) Specific question about the image, default: 'Describe this image in detail.'"
        },
    },
    "respond": {
        "function": lambda browser, **kwargs: {
            "success": True,
            "message": kwargs.get("message", ""),
        },
        "description": "Respond with a final answer to the user.",
        "parameters": {"message": "string"},
    },
}


def get_tools_description() -> str:
    """Get formatted description of all available tools"""
    tools_desc = "Available Tools:\n\n"

    for tool_name, tool_info in TOOLS.items():
        tools_desc += f"- {tool_name}: {tool_info['description']}\n"
        tools_desc += "  Parameters:\n"
        for param_name, param_desc in tool_info["parameters"].items():
            tools_desc += f"    - {param_name}: {param_desc}\n"
        tools_desc += "\n"

    return tools_desc
