from PIL import Image, ImageDraw, ImageFont

# Load image\
path = Path("models_fulltrain")
img = Image.open()
draw = ImageDraw.Draw(img)

# Choose position (pixels from left, pixels from top)
pos = (350, 470)  # adjust based on your image size

# Add text
draw.text(pos, "Epoch", fill="black")  # simple label

# Save new image
img.save("history_with_xlabel.png")
