import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

# 🔹 Define Paths
fine_tuned_model_path = r"C:\Users\rishi\OneDrive\Desktop\Amrita\TAG PROJECT (CASG)\p2 pr2\llama-finetuned\finetuned"  # Update this path
offload_dir = r"C:\Users\rishi\OneDrive\Desktop\Amrita\TAG PROJECT (CASG)\p2 pr2\offload"  # Directory for offloaded layers

# 🔹 Ensure Offload Directory Exists
os.makedirs(offload_dir, exist_ok=True)

# 🔹 Force CPU Offloading with Correct Offload Directory
st.write("🔹 Loading fine-tuned LLaMA model...")

try:
    tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

    model = AutoModelForCausalLM.from_pretrained(
        fine_tuned_model_path,
        device_map="cpu",  # 🔹 Force CPU usage
        torch_dtype=torch.float32,  # 🔹 Use CPU-compatible data type
        offload_folder=offload_dir,  # 🔹 Offload layers to disk
    )

    st.success("✅ LLaMA model loaded successfully!")

except Exception as e:
    st.error(f"❌ Failed to load LLaMA model: {e}")
    model = None  # Prevent further errors

# 🔹 Load BLIP Model for Image Captioning
st.write("🔹 Loading BLIP model for image captioning...")
try:
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    st.success("✅ BLIP model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load BLIP model: {e}")

# 🔹 Streamlit UI
st.title("Ad Script Generator")

theme = st.text_input("🎨 Enter Theme:")
uploaded_image = st.file_uploader("📸 Upload an Image:", type=["jpg", "jpeg", "png"])
caption = st.text_input("📝 Or Enter Caption (if no image):")

# 🔹 Process Image if Uploaded
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate Caption using BLIP
    with st.spinner("🖼 Generating caption from image..."):
        try:
            inputs = blip_processor(image, return_tensors="pt")
            output = blip_model.generate(**inputs)
            caption = blip_processor.decode(output[0], skip_special_tokens=True)
            st.success(f"*Generated Caption:* {caption}")
        except Exception as e:
            st.error(f"❌ Error generating caption: {e}")

# 🔹 Generate Ad Script
if st.button("🎬 Generate Script"):
    if not theme:
        st.error("⚠ Please enter a theme.")
    elif not caption and not uploaded_image:
        st.error("⚠ Please either upload an image or enter a caption.")
    elif model is None:
        st.error("⚠ Model failed to load. Please check the logs above.")
    else:
        with st.spinner("📝 Generating advertisement script..."):
            try:
                input_prompt = f"Generate an advertisement script based on the following details:\nTheme: {theme}\nCaption: {caption}\n"
                inputs = tokenizer(input_prompt, return_tensors="pt", truncation=True)

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_length=500)

                generated_script = tokenizer.decode(outputs[0], skip_special_tokens=True)

                st.subheader("🎭 Generated Script:")
                st.write(generated_script)
            except Exception as e:
                st.error(f"❌ Error generating script: {e}")