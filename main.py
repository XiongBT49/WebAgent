"""
Main Agent Loop - Vision-Language Model Web Agent
Integrates local/cloud LLM with vision capabilities and Playwright for autonomous web browsing
"""

import os
import json
import time
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pathlib import Path

import config
from tools import BrowserTools, TOOLS, get_tools_description


class VLLMWebAgent:
    """Main agent class that coordinates Vision-Language Model and browser tools"""

    def __init__(self):
        """Initialize the agent with the selected LLM provider and vision model."""
        # Select LLM provider based on config
        if config.LLM_PROVIDER == "deepseek":
            print("Using DeepSeek API as LLM provider.")
            if not config.DEEPSEEK_API_KEY:
                raise ValueError(
                    "DEEPSEEK_API_KEY is not set. Please add it to your .env file."
                )
            self.client = OpenAI(
                base_url=config.DEEPSEEK_BASE_URL,
                api_key=config.DEEPSEEK_API_KEY,
            )
            self.model = config.DEEPSEEK_MODEL
        else:  # Default to ollama
            print("Using Ollama as LLM provider.")
            self.client = OpenAI(
                base_url=config.OLLAMA_BASE_URL,
                api_key="EMPTY",  # Ollama doesn't require an API key
            )
            self.model = config.OLLAMA_MODEL

        # Initialize vision model client if enabled
        self.use_vision = config.USE_VISION_MODEL
        if self.use_vision:
            print(f"Vision model enabled: {config.VISION_MODEL_PROVIDER}")
            
            # Ollama Vision Model (Recommended - uses existing Ollama)
            if config.VISION_MODEL_PROVIDER == "ollama":
                print(f"Using Ollama vision model: {config.OLLAMA_VISION_MODEL}")
                self.vision_client = OpenAI(
                    base_url=config.OLLAMA_BASE_URL,
                    api_key="EMPTY"
                )
                self.vision_model = config.OLLAMA_VISION_MODEL
            
            # Local Qwen-VL (requires separate vLLM server)
            elif config.VISION_MODEL_PROVIDER == "qwen-vl-local":
                print(f"Using local Qwen-VL at {config.QWEN_VL_BASE_URL}")
                self.vision_client = OpenAI(
                    base_url=config.QWEN_VL_BASE_URL,
                    api_key="EMPTY"
                )
                self.vision_model = config.QWEN_VL_MODEL
            
            # GPT-4 Vision (Cloud, commented out)
            # elif config.VISION_MODEL_PROVIDER == "gpt-4-vision":
            #     if not config.OPENAI_API_KEY:
            #         print("Warning: OPENAI_API_KEY not set. Disabling vision model.")
            #         self.use_vision = False
            #     else:
            #         self.vision_client = OpenAI(api_key=config.OPENAI_API_KEY)
            #         self.vision_model = config.VISION_MODEL_NAME
            
            else:
                print(f"Warning: Unknown vision provider '{config.VISION_MODEL_PROVIDER}'. Disabling vision.")
                self.use_vision = False
        
        # Initialize browser
        self.browser = BrowserTools()

        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []

        # Execution log
        self.execution_log: List[Dict[str, Any]] = []

        # Track last downloaded PDF filepath (auto-use for extraction)
        self.last_downloaded_pdf: Optional[str] = None

        # Create output directories
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary output directories"""
        for directory in [
            config.OUTPUT_DIR,
            config.SCREENSHOTS_DIR,
            config.PDFS_DIR,
            config.IMAGES_DIR,
            config.LOGS_DIR,
        ]:
            os.makedirs(directory, exist_ok=True)

    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Encode an image file to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def _get_latest_screenshot(self) -> Optional[str]:
        """Get the path to the most recent screenshot"""
        try:
            screenshots = list(Path(config.SCREENSHOTS_DIR).glob("*.png"))
            if not screenshots:
                return None
            latest = max(screenshots, key=lambda p: p.stat().st_mtime)
            return str(latest)
        except Exception as e:
            print(f"Error getting latest screenshot: {e}")
            return None

    def _analyze_image_with_vision(self, image_path: str, question: str) -> Dict[str, Any]:
        """Analyze an image using the vision model"""
        try:
            # Encode image to base64
            base64_image = self._encode_image_to_base64(image_path)
            if not base64_image:
                return {"success": False, "error": f"Failed to encode image: {image_path}"}
            
            # Build vision model prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ]
            
            # Query vision model
            print(f"Analyzing image with vision model: {image_path}")
            response = self.vision_client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
                max_tokens=1000,
                temperature=0.1
            )
            
            analysis = response.choices[0].message.content
            
            return {
                "success": True,
                "image_path": image_path,
                "question": question,
                "analysis": analysis
            }
            
        except Exception as e:
            return {"success": False, "error": f"Vision analysis failed: {str(e)}"}

    def _build_system_prompt(self) -> str:
        """Build the system prompt with tools description"""
        vision_note = ""
        if self.use_vision:
            vision_note = """
**IMPORTANT: VISION CAPABILITIES**
- You can SEE the screenshots! Along with DOM summaries, you receive actual screenshots of the page.
- Use this to identify UI elements that may not have clear text labels (icons, buttons, download symbols, etc.)
- When describing actions, you can reference what you see in the screenshot.
- Example: "I can see a download icon (downward arrow) in the top right of the PDF viewer."
"""
        
        system_prompt = f"""You are a Vision-Language web agent. You can SEE webpages through screenshots AND analyze their structure through DOM summaries.

**YOUR VISION-POWERED WORKFLOW:**
1. **Observe:** You receive BOTH a screenshot AND DOM summary of the current page.
2. **Analyze:** Use the screenshot to understand the visual layout and find UI elements (buttons, icons, links).
3. **Decide:** Based on BOTH visual and structural information, choose the next action.
4. **Act:** Execute one action at a time.

{vision_note}

**HOW TO DOWNLOAD AND EXTRACT PDFs:**
When you need to download a PDF from arXiv:
1. Find the PDF link in search results: `{{"action": "download_pdf", "parameters": {{"pdf_url": "https://arxiv.org/pdf/XXXXX"}}}}`
2. Extract text: `{{"action": "pdf_extract_text", "parameters": {{}}}}`  ← **You can leave pdf_path empty! System auto-uses last download**
3. Extract images: `{{"action": "pdf_extract_images", "parameters": {{}}}}`  ← **Same here, no path needed!**

**AUTOMATIC PATH MANAGEMENT:**
- ✅ The system automatically tracks the last downloaded PDF
- ✅ You DON'T need to specify pdf_path - just use empty parameters {{}}
- ✅ If you do specify a wrong path, the system auto-corrects it
- ✅ This uses Playwright to click download buttons (not HTTP requests)

**EXAMPLE - SIMPLE WORKFLOW:**
```
Step 1: {{"action": "download_pdf", "parameters": {{"pdf_url": "https://arxiv.org/pdf/2511.08246"}}}}
Step 2: {{"action": "pdf_extract_text", "parameters": {{}}}}      ← Empty! System knows which file
Step 3: {{"action": "pdf_extract_images", "parameters": {{}}}}    ← Empty! System knows which file
```

**EXAMPLE WORKFLOW FOR ARXIV:**
1. `goto` → https://arxiv.org/
2. `type` → "Qwen" into search box
3. `press` → Enter
4. Look at specialized_content.search_results (system parses arXiv results for you!)
5. `download_pdf` → Use pdf_url from search results
6. `pdf_extract_text` → No parameters needed! System auto-uses downloaded file
7. `pdf_extract_images` → No parameters needed! System auto-uses downloaded file  
8. **CRITICAL - MANDATORY STEP**: After extracting images, use `analyze_image`:
   - **MUST** look at the "images" array in the pdf_extract_images result
   - **MUST** use the EXACT filepath from the result (e.g., "output/images/downloaded_20251112_082920_page1_full.png")
   - The first rendered page image (page1_full.png) usually contains Figure 1
   - Example: `{{"action": "analyze_image", "parameters": {{"image_path": "output/images/downloaded_20251112_082920_page1_full.png", "question": "Describe Figure 1 in detail, including its purpose, main components, and key findings"}}}}`
   - **DO NOT** make up or guess image paths like "figure_1.png"
   - **DO NOT** skip this step - vision analysis is REQUIRED
9. **ONLY AFTER** successful vision analysis, use `respond` → Write detailed interpretation based on the ACTUAL vision model output

**AVAILABLE TOOLS:**
{get_tools_description()}

**OUTPUT FORMAT (CRITICAL!):**
You MUST output actions in pure JSON format, ONE action per response:
```json
{{"action": "goto", "parameters": {{"url": "https://arxiv.org/"}}}}
```

Or for actions with no parameters:
```json
{{"action": "dom_summary", "parameters": {{}}}}
```

DO NOT add explanatory text before or after the JSON. ONLY output the JSON action.

**RULES:**
- Output ONLY a single JSON object per response
- ALWAYS observe the current state before acting
- Use specialized_content when available (arXiv search results are pre-parsed for you!)
- For PDF operations, the system handles file paths automatically
- If an action fails, try a different approach
- Complete ALL parts of the task: download → extract text/images → **analyze Figure 1 with vision model (MANDATORY)** → respond
- **NEVER skip analyze_image step** - the final response MUST be based on actual vision model output
- **NEVER make up content** - if vision analysis fails, report the failure and try again

Begin your task now!
"""
        return system_prompt

    def _parse_agent_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the agent's response to extract a list of actions.
        Handles both pure JSON and JSON within markdown code blocks.
        """
        actions = []
        
        # First, try to extract JSON from markdown code blocks
        import re
        
        # Pattern 1: JSON code blocks ```json ... ```
        json_blocks = re.findall(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        
        # Pattern 2: Plain code blocks ``` ... ```
        if not json_blocks:
            json_blocks = re.findall(r'```\s*(.*?)\s*```', response, re.DOTALL)
        
        # Pattern 3: Inline JSON objects
        if not json_blocks:
            json_blocks = re.findall(r'\{[^{}]*"action"[^{}]*\}', response, re.DOTALL)
        
        # Try to parse each found block
        for block in json_blocks:
            block = block.strip()
            if not block:
                continue
                
            try:
                # Try to parse as JSON
                parsed = json.loads(block)
                if isinstance(parsed, dict) and "action" in parsed:
                    actions.append(parsed)
                    print(f"[Info] Parsed action: {parsed.get('action')}")
            except json.JSONDecodeError as e:
                print(f"[Warning] Failed to parse JSON block: {block[:100]}... Error: {e}")
                continue
        
        # Fallback: line-by-line parsing for plain JSON
        if not actions:
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line or not line.startswith('{'):
                    continue
                
                try:
                    json_start = line.find('{')
                    if json_start == -1:
                        continue
                    
                    json_str = line[json_start:]
                    
                    # Try to parse
                    try:
                        parsed = json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try to repair incomplete JSON
                        repaired = json_str + '"}}'
                        parsed = json.loads(repaired)
                    
                    if isinstance(parsed, dict) and "action" in parsed:
                        actions.append(parsed)
                        print(f"[Info] Parsed action from line: {parsed.get('action')}")
                        
                except (json.JSONDecodeError, Exception) as e:
                    continue
        
        # If still no actions found, this might be explanatory text only
        # DO NOT treat as respond - let the agent continue
        if not actions:
            print("[Warning] No valid JSON actions found in response. Agent may need clearer instructions.")
            print(f"[Debug] Response preview: {response[:200]}...")
        
        return actions

    def _execute_tool(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return the result with automatic path management"""
        if action not in TOOLS:
            available_tools = ", ".join(TOOLS.keys())
            return {
                "success": False, 
                "error": f"Unknown tool: {action}. Available tools: {available_tools}. Please use one of the available tools."
            }

        try:
            # === AUTOMATIC PDF PATH MANAGEMENT ===
            # Auto-fix pdf_path for extraction tools
            if action in ["pdf_extract_text", "pdf_extract_images"]:
                pdf_path = parameters.get("pdf_path")
                
                # Case 1: No path provided - use last downloaded PDF
                if not pdf_path and self.last_downloaded_pdf:
                    print(f"[Auto-fix] No pdf_path provided, using last downloaded: {self.last_downloaded_pdf}")
                    parameters["pdf_path"] = self.last_downloaded_pdf
                
                # Case 2: Wrong path provided - auto-correct to last downloaded
                elif pdf_path and self.last_downloaded_pdf:
                    # Check if file doesn't exist but we have a recent download
                    if not os.path.exists(pdf_path) and os.path.exists(self.last_downloaded_pdf):
                        print(f"[Auto-fix] File not found: {pdf_path}")
                        print(f"[Auto-fix] Using last downloaded PDF instead: {self.last_downloaded_pdf}")
                        parameters["pdf_path"] = self.last_downloaded_pdf
            
            # Execute the tool with (possibly corrected) parameters
            tool_info = TOOLS[action]
            result = tool_info["function"](self.browser, **parameters)
            
            # Special handling: analyze_image needs vision model
            if action == "analyze_image" and result.get("needs_vision_analysis"):
                if self.use_vision:
                    result = self._analyze_image_with_vision(
                        result.get("image_path"),
                        result.get("question", "Describe this image in detail.")
                    )
                else:
                    result = {
                        "success": False,
                        "error": "Vision model not enabled. Cannot analyze images."
                    }
            
            # Track successful PDF downloads
            if action == "download_pdf" and result.get("success"):
                self.last_downloaded_pdf = result.get("filepath")
                print(f"[Tracked] Last downloaded PDF: {self.last_downloaded_pdf}")
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Error executing {action}: {str(e)}"}

    def _get_current_state(self) -> tuple:
        """
        Get current browser state as context
        Returns: (state_text, screenshot_path)
        """
        if self.browser.page is None:
            return "Browser not initialized. Use 'goto' to navigate to a URL.", None

        try:
            time.sleep(1)  # Wait for page to settle
            
            # Get DOM summary
            dom_result = self.browser.dom_summary()

            # Take screenshot
            screenshot_result = self.browser.screenshot()
            screenshot_path = screenshot_result.get("filepath") if screenshot_result.get("success") else None

            state_info = f"""
Current Browser State:
- URL: {self.browser.page.url}
- Title: {self.browser.page.title()}
- Screenshot saved: {screenshot_result.get("filename", "N/A")}

DOM Summary:
- Links: {len(dom_result.get("summary", {}).get("links", []))} found
- Buttons: {len(dom_result.get("summary", {}).get("buttons", []))} found
- Input fields: {len(dom_result.get("summary", {}).get("inputs", []))} found
"""

            # Add specialized content (arXiv search results)
            summary = dom_result.get("summary", {})
            specialized = summary.get("specialized_content", {})
            
            if specialized:
                if "search_results" in specialized:
                    results = specialized["search_results"]
                    state_info += f"\n**IMPORTANT: Found {len(results)} arXiv search results:**\n\n"
                    for i, result in enumerate(results, 1):
                        state_info += f"{i}. Title: {result['title']}\n"
                        state_info += f"   Authors: {result['authors']}\n"
                        state_info += f"   Date: {result.get('date', 'N/A')}\n"
                        state_info += f"   PDF Link: {result['pdf_href']}\n"
                        state_info += f"   Abstract Link: {result['abstract_href']}\n\n"
                    
                    first_pdf = results[0]['pdf_href'] if results else None
                    if first_pdf:
                        state_info += f"**To download the first PDF, use:**\n"
                        state_info += f'{{\"action\": \"download_pdf\", \"parameters\": {{\"pdf_url\": \"{first_pdf}\"}} }}\n'
                
                if "pdf_link" in specialized:
                    pdf_info = specialized["pdf_link"]
                    state_info += f"\n**PDF Download Available:**\n"
                    state_info += f"  - {pdf_info['text']}: {pdf_info['href']}\n"

            if summary.get("links"):
                state_info += "\nKey Links (first 5):\n"
                for link in summary["links"][:5]:
                    state_info += f"  - {link['text']}: {link['href']}\n"

            if summary.get("inputs"):
                state_info += "\nInput fields:\n"
                for inp in summary["inputs"][:5]:
                    state_info += f"  - {inp['type']}: name='{inp.get('name', '')}' id='{inp.get('id', '')}'\n"

            return state_info, screenshot_path
        except Exception as e:
            return f"Error getting current state: {str(e)}", None

    def run(self, user_goal: str, max_iterations: Optional[int] = None) -> str:
        """
        Main execution loop

        Args:
            user_goal: The user's natural language goal
            max_iterations: Maximum number of iterations (default from config)

        Returns:
            Final response from the agent
        """
        if max_iterations is None:
            max_iterations = config.MAX_ITERATIONS

        print(f"\n{'=' * 80}")
        print(f"STARTING NEW TASK")
        print(f"{'=' * 80}")
        print(f"Goal: {user_goal}")
        print(f"Max iterations: {max_iterations}")
        print(f"{'=' * 80}\n")

        # Initialize conversation with user goal
        self.conversation_history = [
            {"role": "system", "content": self._build_system_prompt()},
            {
                "role": "user",
                "content": f"Please help me accomplish this task: {user_goal}",
            },
        ]

        iteration = 0
        final_response = ""

        try:
            while iteration < max_iterations:
                iteration += 1
                print(f"\n--- Iteration {iteration}/{max_iterations} ---")

                # Get response from the selected LLM
                print(f"Querying {config.LLM_PROVIDER}...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation_history,
                    temperature=config.TEMPERATURE,
                    max_tokens=2000,
                )

                agent_response = response.choices[0].message.content
                print(f"\nAgent response:\n{agent_response}\n")

                # Add to conversation history
                self.conversation_history.append(
                    {"role": "assistant", "content": agent_response}
                )

                # Parse response to get a list of actions
                actions = self._parse_agent_response(agent_response)

                # Flag to check if we should continue to the next iteration
                continue_to_next_iteration = True

                for parsed_action in actions:
                    action = parsed_action.get("action")

                    # Log execution
                    log_entry = {
                        "iteration": iteration,
                        "timestamp": datetime.now().isoformat(),
                        "action": action,
                        "parsed_response": parsed_action,
                    }

                    # Handle action
                    if action == "respond":
                        # Agent is responding to user or task complete
                        message = parsed_action.get("parameters", {}).get("message", agent_response)
                        print(f"\nAgent message: {message}")
                        final_response = message
                        log_entry["status"] = "completed"
                        self.execution_log.append(log_entry)
                        continue_to_next_iteration = False  # Stop after responding
                        break

                    elif action in TOOLS:
                        # Execute tool
                        parameters = parsed_action.get("parameters", {})
                        print(f"Executing tool: {action}")
                        print(f"Parameters: {json.dumps(parameters, indent=2)}")

                        # AUTO-FIX: If agent is trying to extract from PDF with wrong filepath, fix it automatically
                        if action in ["pdf_extract_text", "pdf_extract_images"]:
                            provided_path = parameters.get("pdf_path", "")
                            # Find the most recent successful download
                            correct_filepath = None
                            for log_entry_check in reversed(self.execution_log):
                                if log_entry_check.get("action") == "download_pdf":
                                    log_result = log_entry_check.get("result", {})
                                    if log_result.get("success"):
                                        correct_filepath = log_result.get("filepath")
                                        break
                            
                            # If agent provided wrong path and we know the correct one, auto-fix it
                            if correct_filepath and provided_path != correct_filepath:
                                print(f"[Auto-Fix] Agent provided wrong filepath: {provided_path}")
                                print(f"[Auto-Fix] Correcting to: {correct_filepath}")
                                parameters["pdf_path"] = correct_filepath

                        result = self._execute_tool(action, parameters)
                        log_entry["result"] = result
                        self.execution_log.append(log_entry)
                        print(f"Result: {json.dumps(result, indent=2)}")
                        
                        # ===== NEW: Provide helpful feedback after pdf_extract_images =====
                        if action == "pdf_extract_images" and result.get("success"):
                            images = result.get("images", [])
                            extraction_method = result.get("extraction_method", "unknown")
                            
                            if images:
                                # Check if we got high-quality embedded images or full page renders
                                embedded_images = [img for img in images if img.get("type") == "embedded_image"]
                                full_page_images = [img for img in images if img.get("type") == "rendered_full_page"]
                                
                                if embedded_images:
                                    # IDEAL CASE: We got isolated figure images
                                    first_image_path = embedded_images[0].get("filepath")
                                    hint_message = f"""
[System Hint] ✓ Successfully extracted {len(embedded_images)} high-quality embedded figure(s)!
These are ISOLATED FIGURES (not full pages with text).

You MUST now analyze Figure 1 using the 'analyze_image' tool.
Use this EXACT path: {first_image_path}

Example:
{{"action": "analyze_image", "parameters": {{"image_path": "{first_image_path}", "question": "Describe this figure in detail, including its purpose, main components, and key findings"}}}}

DO NOT skip this step. DO NOT make up image paths.
"""
                                elif full_page_images:
                                    # FALLBACK CASE: Only got full page renders
                                    first_image_path = full_page_images[0].get("filepath")
                                    hint_message = f"""
[System Hint] ⚠ Warning: Could not extract isolated figures from PDF.
Only full page renders are available (which include text + figures mixed together).

The vision model will need to analyze the ENTIRE PAGE and identify Figure 1 within it.
Use this EXACT path: {first_image_path}

Example:
{{"action": "analyze_image", "parameters": {{"image_path": "{first_image_path}", "question": "Find and describe Figure 1 in this page, including its purpose and key findings. Ignore the surrounding text."}}}}

This is not ideal but should work. Proceed with caution.
"""
                                else:
                                    first_image_path = images[0].get("filepath")
                                    hint_message = f"""
[System Hint] Extracted images. Path: {first_image_path}
Analyze using: {{"action": "analyze_image", "parameters": {{"image_path": "{first_image_path}", "question": "Describe Figure 1"}}}}
"""
                                
                                self.conversation_history.append({
                                    "role": "user", 
                                    "content": hint_message
                                })
                                print(hint_message)

                    else:
                        # Unknown action - provide helpful guidance
                        available_tools_list = ", ".join(TOOLS.keys())
                        print(f"Warning: Unknown action '{action}'")
                        
                        # Check if agent is trying to respond without vision analysis
                        has_vision_analysis = False
                        for log_entry_check in self.execution_log:
                            if log_entry_check.get("action") == "analyze_image":
                                if log_entry_check.get("result", {}).get("success"):
                                    has_vision_analysis = True
                                    break
                        
                        # If agent tries to respond without analyzing Figure 1, warn them
                        if action == "respond" and not has_vision_analysis:
                            error_feedback = f"""
ERROR: You are trying to 'respond' but you have NOT successfully analyzed Figure 1 yet!

You MUST:
1. Use 'pdf_extract_images' to extract images from the PDF (if not done already)
2. Use 'analyze_image' with the EXACT image path from step 1 result
3. ONLY after successful vision analysis, use 'respond' with content based on actual vision output

DO NOT make up content. DO NOT skip vision analysis.
"""
                        else:
                            # Add helpful feedback to guide agent
                            error_feedback = f"""
ERROR: Tool '{action}' does not exist!

Available tools: {available_tools_list}

HINT: Based on your goal, you might want to use:
- To extract text from PDF: use 'pdf_extract_text'
- To extract images from PDF: use 'pdf_extract_images'  
- To analyze an image: First extract it with 'pdf_extract_images', then the system will show you the image in the next observation
- To respond with final answer: use 'respond'
"""
                        self.conversation_history.append({
                            "role": "user",
                            "content": error_feedback
                        })
                        
                        log_entry["status"] = "unknown_action"
                        log_entry["error"] = f"Tool '{action}' does not exist"
                        self.execution_log.append(log_entry)

                # After executing all actions, decide what to do next
                if not continue_to_next_iteration:
                    break  # Exit the main while loop

                # Get current state after all actions in the plan are done
                state_text, screenshot_path = self._get_current_state()
                
                # Build feedback message with vision support
                if self.use_vision and screenshot_path:
                    # Encode screenshot for vision model
                    image_base64 = self._encode_image_to_base64(screenshot_path)
                    if image_base64:
                        # Use vision model format (GPT-4V style)
                        feedback_message = {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"""
All actions in the previous plan have been executed.
Result and current state:
{state_text}

[You can see the current page in the attached screenshot]

What's the next step?
"""
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                        self.conversation_history.append(feedback_message)
                    else:
                        # Fallback if image encoding fails
                        feedback = f"""
All actions in the previous plan have been executed.
Result and current state:
{state_text}

What's the next step?
"""
                        self.conversation_history.append({"role": "user", "content": feedback})
                else:
                    # No vision model or screenshot
                    feedback = f"""
All actions in the previous plan have been executed.
Result and current state:
{state_text}

What's the next step?
"""
                    self.conversation_history.append({"role": "user", "content": feedback})

                # Small delay to avoid overwhelming the system
                time.sleep(0.5)

            # This part is outside the loop, so we remove the old log saving
            # self.execution_log.append(log_entry)

            if iteration >= max_iterations:
                final_response = f"Maximum iterations ({max_iterations}) reached. Task may be incomplete."
                print(f"\n{final_response}")

            return final_response

        except KeyboardInterrupt:
            print("\n\nTask interrupted by user.")
            self._save_execution_log()
            return "Task interrupted by user."

        except Exception as e:
            print(f"\n\nError during execution: {str(e)}")
            import traceback

            traceback.print_exc()
            self._save_execution_log()
            return f"Error: {str(e)}"

        finally:
            # Save execution log before cleanup
            self._save_execution_log()
            
            # Cleanup
            print("\nCleaning up browser resources...")
            self.browser.cleanup()

    def _save_execution_log(self):
        """Save execution log to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"execution_log_{timestamp}.json"
        log_filepath = os.path.join(config.LOGS_DIR, log_filename)

        log_data = {
            "timestamp": timestamp,
            "conversation_history": self.conversation_history,
            "execution_log": self.execution_log,
        }

        with open(log_filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        print(f"\nExecution log saved to: {log_filepath}")


def main():
    """Main entry point"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py '<your task description>'")
        print("\nExample:")
        print(
            '  python main.py "Find the most recent technical report about Qwen, then interpret Figure 1"'
        )
        sys.exit(1)

    user_goal = sys.argv[1]

    agent = VLLMWebAgent()
    result = agent.run(user_goal)

    print(f"\n{'=' * 80}")
    print("TASK COMPLETED")
    print(f"{'=' * 80}")
    print(f"Final result: {result}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
