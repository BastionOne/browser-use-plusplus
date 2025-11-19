#!/usr/bin/env python3
"""
Agent Snapshot Viewer Generator

Generates a static HTML page displaying agent execution steps with screenshots and data.
Usage: python generate_viewer.py <folder_path>
"""

import json
import base64
import sys
import webbrowser
from pathlib import Path
from typing import Dict, List, Any

from browser_use_plusplus.src.prompts.planv4 import PlanItem


def load_snapshot_data(folder_path: Path) -> Dict[int, Dict[str, Any]]:
    """Load and parse snapshot.json file keyed by step number."""
    snapshot_file = folder_path / "snapshots.json"
    
    if not snapshot_file.exists():
        raise FileNotFoundError(f"snapshots.json not found in {folder_path}")
    
    with open(snapshot_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def get_screenshots(folder_path: Path) -> Dict[int, Path]:
    """Find all screenshots in format step_n.png and return dict keyed by step number."""
    screenshots = {}
    
    for png_file in folder_path.glob("screenshots/*.png"):
        # Extract step number from filename
        try:
            step_num = png_file.stem.split('_')[1]
            screenshots[step_num] = png_file
        except (IndexError, ValueError):
            print(f"Warning: Skipping malformed filename: {png_file.name}")
            continue
    return screenshots


def encode_image_to_base64(image_path: Path) -> str:
    """Encode PNG image to base64 data URI."""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    base64_data = base64.b64encode(image_data).decode('utf-8')
    return f"data:image/png;base64,{base64_data}"


def format_plan_item(plan: Dict[str, Any]) -> str:
    """Format a plan item as HTML."""
    if not plan:
        return "<em>None</em>"
    
    html = "<div class='plan-item'>"
    for key, value in plan.items():
        if isinstance(value, list):
            value = "<br>".join([f"• {item}" for item in value])
        elif isinstance(value, dict):
            value = json.dumps(value, indent=2)
        html += f"<strong>{key}:</strong> {value}<br>"
    html += "</div>"
    return html


def format_plan_list(plans: List[str]) -> str:
    """Format a list of plan items as HTML."""
    if not plans:
        return "<em>None</em>"
    
    html = "<div class='plan-list'>"
    for i, plan in enumerate(plans, 1):
        html += f"<div class='plan-list-item'><strong>Plan {i}:</strong><br>{plan}</div>"
    html += "</div>"
    return html


def generate_html(snapshots: Dict[int, Dict[str, Any]], screenshots: Dict[int, Path]) -> str:
    """Generate the complete HTML document."""
    
    # Get sorted step numbers
    all_steps = sorted(set(int(k) for k in snapshots["snapshots"].keys()) | set(int(k) for k in screenshots.keys()))
    all_steps = [str(step) for step in all_steps]

    # Generate step sections
    step_sections = []
    
    for step_num in all_steps:
        snapshot = snapshots["snapshots"].get(step_num, {})
        screenshot_path = screenshots.get(step_num)
        
        # Encode screenshot if available
        screenshot_html = ""
        if screenshot_path:
            img_data = encode_image_to_base64(screenshot_path)
            screenshot_html = f'<img src="{img_data}" alt="Step {step_num} Screenshot">'
        else:
            screenshot_html = '<div class="no-screenshot">No screenshot available</div>'
        
        # Extract agent data
        completed_plans = [
            PlanItem.model_validate(plan).description for plan in snapshot.get('completed_plans', [])
        ]
        
        curr_url = snapshot.get('curr_url', 'N/A')

        agent_state = snapshot.get("bu_agent_state", {}).get("state", {})
        last_output = agent_state.get("last_model_output", {})
        
        evaluation = last_output.get("evaluation_previous_goal", "N/A")
        next_goal = last_output.get("next_goal", "N/A")


        # Build data section
        data_html = f"""
        <div class="data-field">
            <div class="data-label">Current URL:</div>
            <div class="data-value url">{curr_url}</div>
        </div>
        
        <div class="data-field">
            <div class="data-label">Evaluation of Previous Goal:</div>
            <div class="data-value">{evaluation}</div>
        </div>
        
        <div class="data-field">
            <div class="data-label">Next Goal:</div>
            <div class="data-value">{next_goal}</div>
        </div>
                
        <div class="data-field">
            <div class="data-label">Completed Plans:</div>
            <div class="data-value">{format_plan_list(completed_plans)}</div>
        </div>
        """
        
        step_section = f"""
        <div class="step-container" id="step-{step_num}">
            <h2 class="step-header">Step {step_num}</h2>
            <div class="step-content">
                <div class="screenshot-panel">
                    {screenshot_html}
                </div>
                <div class="data-panel">
                    {data_html}
                </div>
            </div>
        </div>
        """
        
        step_sections.append(step_section)
    
    # Complete HTML document
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Execution Viewer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        
        .header {{
            background: #2c3e50;
            color: white;
            padding: 2rem;
            text-align: center;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}
        
        .header p {{
            opacity: 0.9;
            font-size: 1rem;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        .step-container {{
            background: white;
            border-radius: 8px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .step-header {{
            background: #34495e;
            color: white;
            padding: 1rem 1.5rem;
            font-size: 1.5rem;
            font-weight: 600;
        }}
        
        .step-content {{
            display: flex;
            gap: 2rem;
            padding: 1.5rem;
        }}
        
        .screenshot-panel {{
            flex: 0 0 70%;
            max-width: 70%;
        }}
        
        .screenshot-panel img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
            border: 1px solid #ddd;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .no-screenshot {{
            background: #ecf0f1;
            border: 2px dashed #bdc3c7;
            border-radius: 4px;
            padding: 4rem 2rem;
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
        }}
        
        .data-panel {{
            flex: 1;
            overflow-y: auto;
            max-height: 800px;
        }}
        
        .data-field {{
            margin-bottom: 1.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .data-field:last-child {{
            border-bottom: none;
        }}
        
        .data-label {{
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 0.5rem;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .data-value {{
            color: #555;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        
        .data-value.url {{
            color: #3498db;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
        }}
        
        .plan-item {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 4px;
            margin-top: 0.5rem;
            font-size: 0.9rem;
        }}
        
        .plan-list {{
            margin-top: 0.5rem;
        }}
        
        .plan-list-item {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 0.75rem;
            border-left: 3px solid #3498db;
        }}
        
        .plan-list-item:last-child {{
            margin-bottom: 0;
        }}
        
        /* Responsive design */
        @media (max-width: 1200px) {{
            .step-content {{
                flex-direction: column;
            }}
            
            .screenshot-panel {{
                flex: 1;
                max-width: 100%;
            }}
            
            .data-panel {{
                max-height: none;
            }}
        }}
        
        /* Navigation */
        .nav {{
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            display: flex;
            gap: 0.5rem;
            z-index: 99;
        }}
        
        .nav button {{
            background: #3498db;
            color: white;
            border: none;
            padding: 1rem 1.5rem;
            border-radius: 50px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
            transition: all 0.3s ease;
        }}
        
        .nav button:hover {{
            background: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(52, 152, 219, 0.4);
        }}
        
        .nav button:active {{
            transform: translateY(0);
        }}
        
        .nav button:disabled {{
            background: #95a5a6;
            cursor: not-allowed;
            box-shadow: none;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Agent Execution Viewer</h1>
        <p>Viewing {len(all_steps)} execution steps</p>
    </div>
    
    <div class="container">
        {''.join(step_sections)}
    </div>
    
    <div class="nav">
        <button onclick="scrollToPrevious()">← Previous</button>
        <button onclick="scrollToNext()">Next →</button>
    </div>
    
    <script>
        let currentStep = 0;
        const steps = {json.dumps(all_steps)};
        
        function scrollToStep(stepNum) {{
            const element = document.getElementById(`step-${{stepNum}}`);
            if (element) {{
                element.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
            }}
        }}
        
        function scrollToNext() {{
            if (currentStep < steps.length - 1) {{
                currentStep++;
                scrollToStep(steps[currentStep]);
            }}
        }}
        
        function scrollToPrevious() {{
            if (currentStep > 0) {{
                currentStep--;
                scrollToStep(steps[currentStep]);
            }}
        }}
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {{
                e.preventDefault();
                scrollToNext();
            }} else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {{
                e.preventDefault();
                scrollToPrevious();
            }}
        }});
        
        // Update current step based on scroll position
        window.addEventListener('scroll', () => {{
            const windowHeight = window.innerHeight;
            const scrollY = window.scrollY;
            
            for (let i = 0; i < steps.length; i++) {{
                const element = document.getElementById(`step-${{steps[i]}}`);
                if (element) {{
                    const rect = element.getBoundingClientRect();
                    if (rect.top <= windowHeight / 3) {{
                        currentStep = i;
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
    """
    
    return html


def main():
    """Main execution function."""
    if len(sys.argv) != 2:
        print("Usage: python generate_viewer.py <folder_path>")
        sys.exit(1)
    
    folder_path = Path(sys.argv[1])
    
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        sys.exit(1)
    
    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory")
        sys.exit(1)
    
    print(f"Loading data from: {folder_path}")
    
    # Load snapshot data
    try:
        snapshots = load_snapshot_data(folder_path)
        print(f"✓ Loaded {len(snapshots)} snapshots")
    except Exception as e:
        print(f"Error loading snapshot.json: {e}")
        sys.exit(1)
    
    # Load screenshots
    screenshots = get_screenshots(folder_path)
    print(f"✓ Found {len(screenshots)} screenshots")
    
    # Generate HTML
    print("Generating HTML...")
    html_content = generate_html(snapshots, screenshots)
    
    # Save HTML file
    output_file = folder_path / "agent_viewer.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Generated: {output_file}")
    
    # Open in browser
    print("Opening in browser...")
    webbrowser.open(f"file://{output_file.absolute()}")
    print("Done!")


if __name__ == "__main__":
    main()