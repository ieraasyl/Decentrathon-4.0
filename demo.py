import gradio as gr
import requests

API_URL = "http://localhost:8000/predict"
TRUST_URL = "http://localhost:8000/trust-score"

def analyze_image(image_path):
    try:
        with open(image_path, "rb") as f:
            files = {"file": ("upload.jpg", f, "image/jpeg")}
            resp = requests.post(TRUST_URL, files=files)

        if resp.status_code == 200:
            result = resp.json()
            return (
                result.get("predicted_class", "N/A"),
                result.get("confidence", 0.0),
                f"{result.get('trust_score', 0)}%",
                result.get("explanation", "")
            )
        else:
            return ("Error", 0.0, "N/A", resp.text)
    except Exception as e:
        return ("Error", 0.0, "N/A", str(e))

with gr.Blocks() as demo:
    gr.Markdown("## 🚘 inDrive AI Image Analysis Demo")
    gr.Markdown("Upload a photo of a car to check **cleanliness & damage status**.\n\n"
                "This uses our FastAPI backend model and returns a **Trust Score**.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload Car Photo")
            submit_btn = gr.Button("Analyze")
        with gr.Column():
            label_out = gr.Textbox(label="Predicted Class")
            conf_out = gr.Textbox(label="Confidence")
            trust_out = gr.Textbox(label="Trust Score")
            explanation_out = gr.Textbox(label="Explanation")

    submit_btn.click(
        analyze_image,
        inputs=image_input,
        outputs=[label_out, conf_out, trust_out, explanation_out]
    )

if __name__ == "__main__":
    demo.launch()