# Image Classifier

A small Streamlit demo that analyzes uploaded PNG and JPEG images locally. Its built-in labels are heuristic descriptions based on colour, brightness, saturation, and pixel variation; they are not trained-model predictions.

An optional button can send the selected image to Hugging Face's ViT inference endpoint. That request is made only after you enter a Hugging Face token and click **Classify with ViT**. The app does not save the token.

## Run

```powershell
uv sync --no-config
uv run --no-config streamlit run main.py
```

Upload a PNG or JPEG file. The local results show image statistics and heuristic labels. To use the optional ViT result, provide a valid Hugging Face token in the interface, then click the button.

## Notes

- Local analysis runs entirely on your computer.
- The optional Hugging Face action requires internet access and shares the selected image with that external service.
- Heuristic confidence values are normalized rule scores, not calibrated probabilities.
