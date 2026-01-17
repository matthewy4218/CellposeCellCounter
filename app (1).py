import gradio as gr
import spaces
from cellpose import models
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tempfile
from PIL import Image
import io
from huggingface_hub import hf_hub_download

HF_REPO_ID = "myang4218/cellposemodel"
MODEL_OPTIONS = {
    "Hemocytometer Model": "hemocytometermodel.npy",
    "General Model": "generalmodel.npy"
}

loaded_models = {}

def extract_region_from_editor(editor_data):
    """Extract the selected region """
    if editor_data is None:
        return None, None
    
    if isinstance(editor_data, dict):
        background = editor_data.get('background')
        layers = editor_data.get('layers', [])
        
        if background is None:
            return None, None
            
        background_np = np.array(background)
        
        if layers and len(layers) > 0:
            selection_layer = layers[0]
            selection_np = np.array(selection_layer)
            
            if len(selection_np.shape) == 3:
                if selection_np.shape[2] == 4:  # RGBA
                    mask = selection_np[:, :, 3] > 0
                else:  # RGB
                    mask = np.any(selection_np > 0, axis=2)
            else:
                mask = selection_np > 0
            
            coords = np.where(mask)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                pad = 5
                h, w = background_np.shape[:2]
                y_min = max(0, y_min - pad)
                y_max = min(h, y_max + pad)
                x_min = max(0, x_min - pad)
                x_max = min(w, x_max + pad)
                
                region = background_np[y_min:y_max+1, x_min:x_max+1]
                return region, (x_min, y_min, x_max, y_max)
        
        return background_np, None
    
    else:
        if hasattr(editor_data, 'size'):
            image_np = np.array(editor_data)
            return image_np, None
        else:
            return None, None

def classify_cells_by_blueness(image_np, masks, blue_threshold):
   
    
    # Ensure image_np is RGB 
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    #convert
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
    # Calculate blueness index for each pixel
    hue = hsv[:, :, 0].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    
    # Hue score
    hue_distance = np.minimum(np.abs(hue - 115), 180 - np.abs(hue - 115))
    hue_score = np.maximum(0, 1 - hue_distance / 65)  # 65 good blue range
    
    # Combine hue proximity with saturation intensity
    blueness = hue_score * (saturation / 255.0)
    
    # Convert threshold from 0-100 to 0-1 scale
    threshold = blue_threshold / 100.0
    
    # Get unique cell IDs (excluding background)
    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids > 0]  # Remove background (0)
    
    dead_cells = []
    alive_cells = []
    
    # Classify each cell
    for cell_id in cell_ids:
        cell_mask = (masks == cell_id)
        
        # Calculate average blueness for this cell
        cell_blueness = np.mean(blueness[cell_mask])
        
        # Classify based on threshold
        if cell_blueness > threshold:
            dead_cells.append(cell_id)
        else:
            alive_cells.append(cell_id)
    
    # Create colored overlay
    overlay = image_np.copy().astype(np.float32) 
    
    # Color dead cells red, alive cells green
    for cell_id in dead_cells:
        cell_mask = (masks == cell_id)
        overlay[cell_mask] = [255, 0, 0]  # Red for dead
    
    for cell_id in alive_cells:
        cell_mask = (masks == cell_id)
        overlay[cell_mask] = [0, 255, 0]  # Green for alive
    
    # Blend with original image
    alpha = 0.4
    final_overlay = (1 - alpha) * image_np.astype(np.float32) + alpha * overlay
    final_overlay = np.clip(final_overlay, 0, 255).astype(np.uint8)
    
    return len(dead_cells), len(alive_cells), final_overlay
    
def measure_confluency(masks, image_np):
    """Calculate the percentage of image area covered by cells"""
    tot_pixels = image_np.shape[0] * image_np.shape[1]
    cell_pixels = np.count_nonzero(masks)
    confluency = cell_pixels / tot_pixels * 100
    return confluency


@spaces.GPU
def run_segmentation_editor(editor_data, model_choice):
    """
    Runs cell segmentation using ImageEditor data.
    Returns initial segmentation overlay, counts, confluency, and also masks/image for state.
    """
    try:
        model_filename = MODEL_OPTIONS[model_choice]
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=model_filename)
        
        if model_filename in loaded_models:
            model = loaded_models[model_filename]
        else:
            model = models.CellposeModel(gpu=True, pretrained_model=model_path)
            loaded_models[model_filename] = model
        
        region_np, region_coords = extract_region_from_editor(editor_data)
        
        if region_np is None:
            return 0, None, f"No image provided.", gr.update(visible=False), None, None, 0.0
        
        # Resize large images to prevent crashes
        max_size = 1024 # Don't fuck with this
        if region_np.shape[0] > max_size or region_np.shape[1] > max_size:
            h, w = region_np.shape[:2]
            if h > w:
                new_h, new_w = max_size, int(w * max_size / h)
            else:
                new_h, new_w = int(h * max_size / w), max_size
            region_np = cv2.resize(region_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Process image format to RGB
        if len(region_np.shape) == 2:
            processed_image_np = cv2.cvtColor(region_np, cv2.COLOR_GRAY2RGB)
        elif len(region_np.shape) == 3 and region_np.shape[2] == 4:
            processed_image_np = cv2.cvtColor(region_np, cv2.COLOR_RGBA2RGB)
        else:
            processed_image_np = region_np
        
        # Run Cellpose segmentation
        masks, flows, styles = model.eval(processed_image_np, diameter=None, channels=[0, 0])
        
        cell_count = len(np.unique(masks)) - 1
        confluency = measure_confluency(masks, processed_image_np)
        
        # Create a basic segmentation overlay (without viability)
        segmentation_overlay = processed_image_np.copy().astype(np.float32)
        if masks.max() > 0:
            np.random.seed(42) # For consistent random colors
            colors = np.random.randint(0, 255, size=(masks.max() + 1, 3))
            colors[0] = [0, 0, 0] # Background color
            colored_mask = colors[masks]
            alpha = 0.4
            segmentation_overlay = (1 - alpha) * segmentation_overlay + alpha * colored_mask
        segmentation_overlay = np.clip(segmentation_overlay, 0, 255).astype(np.uint8)
        
        info_msg = f"Segmentation complete! Found {cell_count} cells.\n"
        info_msg += f"Confluency: {confluency:.1f}%\n"
        if region_coords:
            info_msg += f"Processed region: {region_coords[0]},{region_coords[1]} to {region_coords[2]},{region_coords[3]}\n"
        info_msg += f"Now adjust the Blue Threshold for viability assessment."
                
        # Return initial segmentation display and state variables
        return cell_count, Image.fromarray(segmentation_overlay), info_msg, gr.update(visible=True), masks, processed_image_np, confluency
            
    except Exception as e:
        return 0, None, f"Error during segmentation: {str(e)}", gr.update(visible=False), None, None, 0.0

@spaces.GPU
def run_segmentation_manual(image, model_choice, crop_coords):
    """
    Runs cell segmentation using manual image input and coordinates.
    Returns initial segmentation overlay, counts, confluency, and also masks/image for state.
    """
    if image is None:
        return 0, None, "No image provided", gr.update(visible=False), None, None, 0.0
        
    try:
        model_filename = MODEL_OPTIONS[model_choice]
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=model_filename)
                
        if model_filename in loaded_models:
            model = loaded_models[model_filename]
        else:
            model = models.CellposeModel(gpu=True, pretrained_model=model_path)
            loaded_models[model_filename] = model
                
        image_np = np.array(image)
        
        # Resize large images to prevent crashes
        max_size = 1024 # Don't fuck with this
        if image_np.shape[0] > max_size or image_np.shape[1] > max_size:
            h, w = image_np.shape[:2]
            if h > w:
                new_h, new_w = max_size, int(w * max_size / h)
            else:
                new_h, new_w = int(h * max_size / w), max_size
            image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
        # Apply crop if coordinates are provided
        if crop_coords and crop_coords.strip():
            try:
                coords = [int(x.strip()) for x in crop_coords.split(',')]
                if len(coords) == 4:
                    x_min, y_min, x_max, y_max = coords
                    h, w = image_np.shape[:2]
                    x_min = max(0, min(x_min, w-1))
                    y_min = max(0, min(y_min, h-1))
                    x_max = max(x_min+1, min(x_max, w))
                    y_max = max(y_min+1, min(y_max, h))
                                        
                    processed_image_np = image_np[y_min:y_max, x_min:x_max]
                else:
                    processed_image_np = image_np # No valid crop, use original
            except ValueError:
                processed_image_np = image_np # Invalid crop, use original
        else:
            processed_image_np = image_np # No crop coords, use original
        
        # Process image format to RGB
        if len(processed_image_np.shape) == 2:
            processed_image_np = cv2.cvtColor(processed_image_np, cv2.COLOR_GRAY2RGB)
        elif len(processed_image_np.shape) == 3 and processed_image_np.shape[2] == 4:
            processed_image_np = cv2.cvtColor(processed_image_np, cv2.COLOR_RGBA2RGB)
        
        # Run Cellpose
        masks, flows, styles = model.eval(processed_image_np, diameter=None, channels=[0, 0])
        
        cell_count = len(np.unique(masks)) - 1    
        confluency = measure_confluency(masks, processed_image_np)
        
        # Create a basic segmentation overlay
        segmentation_overlay = processed_image_np.copy().astype(np.float32)
        if masks.max() > 0:
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(masks.max() + 1, 3))
            colors[0] = [0, 0, 0]                        
            colored_mask = colors[masks]
            alpha = 0.4
            segmentation_overlay = (1 - alpha) * segmentation_overlay + alpha * colored_mask
        segmentation_overlay = np.clip(segmentation_overlay, 0, 255).astype(np.uint8)
                
        info_msg = f"Segmentation complete! Found {cell_count} cells.\n"
        info_msg += f"Confluency: {confluency:.1f}%\n"
        if crop_coords:
            info_msg += f"Processed with coordinates: {crop_coords}\n"
        info_msg += f"Now adjust the Blue Threshold for viability assessment."
                
        return cell_count, Image.fromarray(segmentation_overlay), info_msg, gr.update(visible=True), masks, processed_image_np, confluency
            
    except Exception as e:
        return 0, None, f"Error during segmentation: {str(e)}", gr.update(visible=False), None, None, 0.0

def update_viability_realtime(blue_threshold, stored_masks, stored_image_np):
    """
    Updates viability assessment in real-time based on blue threshold.
    Takes stored masks and image_np from state.
    """
    if stored_masks is None or stored_image_np is None:
        return None, 0, 0, 0.0, "Please run segmentation first."
    
    try:
        dead_count, alive_count, viability_overlay_np = classify_cells_by_blueness(
            stored_image_np, stored_masks, blue_threshold
        )
        
        total_count = alive_count + dead_count
        viability_percent = (alive_count / total_count * 100) if total_count > 0 else 0.0
        confluency = measure_confluency(stored_masks, stored_image_np)
        
        overlay_image = Image.fromarray(viability_overlay_np)
        info_msg = f"Total cells: {total_count}\nLive (green): {alive_count}\nDead (red): {dead_count}\n"
        info_msg += f"Viability: {viability_percent:.1f}%\nConfluency: {confluency:.1f}%\nBlue threshold: {blue_threshold}%"
        
        return overlay_image, alive_count, dead_count, viability_percent, info_msg
            
    except Exception as e:
        return None, 0, 0, 0.0, f"Error updating viability: {str(e)}"

# PWA Head HTML - Updated for nested zip structure
pwa_head = """
<link rel="manifest" href="/file=static.zip/static/manifest.json">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="Cellpose Cell Counter">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="apple-touch-icon" href="/file=favicon.png">
<link rel="icon" type="image/png" sizes="192x192" href="/file=favicon.png">
<script>
if ('serviceWorker' in navigator) {
  window.addEventListener('load', function() {
    navigator.serviceWorker.register('/file=static.zip/static/service-worker.js')
      .then(function(registration) {
        console.log('ServiceWorker registration successful');
      }, function(err) {
        console.log('ServiceWorker registration failed: ', err);
      });
  });
}
</script>
"""

# Create the Gradio interface
with gr.Blocks(
    title="CellposeCellCounter", 
    theme=gr.themes.Soft(),
    head=pwa_head  # Add PWA head elements
) as demo:
    gr.Markdown("# CellposeCellCounter")
    gr.Markdown("For accurate cell confluency, crop the image to display only desired area.")
        
    # Define State components to store masks and image data across function calls
    masks_state = gr.State(value=None)
    image_state = gr.State(value=None)

    with gr.Tab("Image Editor (Draw Selection)"):
        gr.Markdown("### Draw selection and run segmentation")
                
        with gr.Row():
            with gr.Column():
                image_editor = gr.ImageEditor(
                    label="Draw selection on image",
                    type="pil",
                    brush=gr.Brush(colors=["#ff0000"], color_mode="fixed", default_size=20),
                    eraser=gr.Eraser(default_size=20)
                )
                model_dropdown1 = gr.Dropdown(
                    choices=list(MODEL_OPTIONS.keys()),
                    label="Select Model",
                    value="Hemocytometer Model"
                )
                segment_btn1 = gr.Button("🔬 Run Segmentation", variant="primary", size="lg")
                            
            with gr.Column():
                cell_count_output1 = gr.Number(label="Total Cells Detected", precision=0)
                confluency_output1 = gr.Number(label="Confluency (%)", precision=1)
                overlay_output1 = gr.Image(type="pil", label="Segmentation Result")
                info_output1 = gr.Textbox(label="Processing Info", lines=4)
                
        # Viability Assessment Section
        with gr.Group(visible=False) as viability_section1:
            gr.Markdown("### Viability Assessment (Trypan Blue)")
            gr.Markdown("Adjust the threshold to classify cells as live (green) or dead (red).")
                        
            with gr.Row():
                with gr.Column():
                    blue_threshold1 = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=25,
                        step=1,
                        label="Blue Threshold (%)",
                        info="Higher values = more selective for blue cells"
                    )
                                    
                with gr.Column():
                    live_count_output1 = gr.Number(label="Live Cells (Green)", precision=0)
                    dead_count_output1 = gr.Number(label="Dead Cells (Red)", precision=0)
                        
            viability_overlay1 = gr.Image(type="pil", label="Viability Assessment (Green=Live, Red=Dead)")
            viability_percent_output1 = gr.Number(label="Viability (%)", precision=1)
            viability_info1 = gr.Textbox(label="Analysis Results", lines=5)
                
        # Event handlers
        # segment_cells now returns masks and image_np which are stored in masks_state and image_state
        segment_btn1.click(
            fn=run_segmentation_editor,
            inputs=[image_editor, model_dropdown1],
            outputs=[cell_count_output1, overlay_output1, info_output1, viability_section1, masks_state, image_state, confluency_output1]
        ).then(  # Chain the initial viability assessment after segmentation
            fn=update_viability_realtime,
            inputs=[blue_threshold1, masks_state, image_state], # Pass stored state as inputs
            outputs=[viability_overlay1, live_count_output1, dead_count_output1, viability_percent_output1, viability_info1]
        )
                
        # Slider changes update viability in real-time
        blue_threshold1.change(
            fn=update_viability_realtime,
            inputs=[blue_threshold1, masks_state, image_state],
            outputs=[viability_overlay1, live_count_output1, dead_count_output1, viability_percent_output1, viability_info1]
        )

    with gr.Tab("Manual Coordinates"):
        gr.Markdown("### Upload image and run segmentation")
                
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Microscopy Image")
                model_dropdown2 = gr.Dropdown(
                    choices=list(MODEL_OPTIONS.keys()),
                    label="Select Model",
                    value="Hemocytometer Model"
                )
                coord_input = gr.Textbox(
                    label="Crop Coordinates (optional)",
                    placeholder="e.g., 100,100,400,400",
                    info="Format: x_min,y_min,x_max,y_max"
                )
                segment_btn2 = gr.Button("🔬 Run Segmentation", variant="primary", size="lg")
                            
            with gr.Column():
                cell_count_output2 = gr.Number(label="Total Cells Detected", precision=0)
                confluency_output2 = gr.Number(label="Confluency (%)", precision=1)
                overlay_output2 = gr.Image(type="pil", label="Segmentation Result")
                info_output2 = gr.Textbox(label="Processing Info", lines=4)
                
        # Viability Assessment Section
        with gr.Group(visible=False) as viability_section2:
            gr.Markdown("### Viability Assessment (Trypan Blue)")
            gr.Markdown("Adjust the threshold to classify cells as live (green) or dead (red).")
                        
            with gr.Row():
                with gr.Column():
                    blue_threshold2 = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=25,
                        step=1,
                        label="Blue Threshold (%)",
                        info="Higher values = more selective for blue cells"
                    )
                                    
                with gr.Column():
                    live_count_output2 = gr.Number(label="Live Cells (Green)", precision=0)
                    dead_count_output2 = gr.Number(label="Dead Cells (Red)", precision=0)
                        
            viability_overlay2 = gr.Image(type="pil", label="Viability Assessment (Green=Live, Red=Dead)")
            viability_percent_output2 = gr.Number(label="Viability (%)", precision=1)
            viability_info2 = gr.Textbox(label="Analysis Results", lines=5)
                
        # Event handlers
        segment_btn2.click(
            fn=run_segmentation_manual,
            inputs=[image_input, model_dropdown2, coord_input],
            outputs=[cell_count_output2, overlay_output2, info_output2, viability_section2, masks_state, image_state, confluency_output2]
        ).then(  # Chain the initial viability assessment after segmentation
            fn=update_viability_realtime,
            inputs=[blue_threshold2, masks_state, image_state], 
            outputs=[viability_overlay2, live_count_output2, dead_count_output2, viability_percent_output2, viability_info2]
        )
                
        # Slider changes update viability in real-time
        blue_threshold2.change(
            fn=update_viability_realtime,
            inputs=[blue_threshold2, masks_state, image_state],
            outputs=[viability_overlay2, live_count_output2, dead_count_output2, viability_percent_output2, viability_info2]
        )

    # Instructions
    with gr.Accordion("Instructions", open=False):
        gr.Markdown("""
        ### How to use:
                
        1. **Upload and Segment**:
            - Upload your microscopy image.
            - Select a Cellpose model (e.g., "Hemocytometer Model" for blood cells).
            - Draw a selection region using the Image Editor, or specify coordinates manually.
            - Click "Run Segmentation".
                
        2. **Analysis Results**:
            - **Cell Count**: Total number of detected cells
            - **Confluency**: Percentage of image area covered by cells (useful for assessing cell density). Note that cell confluency is calculated per the entire area of the image input.
                
        3. **Real-time Viability Assessment (Trypan Blue)**:
            - After segmentation, the viability section will become visible.
            - This tool is specifically designed for **Trypan Blue stained images**, where dead cells appear blue.
            - Adjust the **"Blue Threshold (%)"** slider in real-time. As you change it, the green (live) and red (dead) classification on the overlay will update.
            - **Lower values (e.g., 10-20%)** are more sensitive and will classify more cells as blue/dead.
            - **Higher values (e.g., 30-50%)** are more selective and will only classify strongly blue cells as dead.
            - Green cells = Live, Red cells = Dead.
                
        4. **Interpreting Results**:
            - The app calculates and displays the total, live, and dead cell counts, along with the viability percentage and confluency.
            - **Confluency** helps assess how densely packed your cells are, which is important for cell culture monitoring.
        """)

if __name__ == "__main__":
    demo.launch()
