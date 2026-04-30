from PIL import Image, ImageDraw, ImageFont
import os

# -------- paths --------
fig6_path = "outputs/fig6/fake_samples_8000_fig6.png"
fig7_path = "outputs/fig7/fake_samples_8000_fig7.png"
fig8_path = "outputs/fig8/fake_samples_8000_fig8.png"

print("Fig6 exists:", os.path.exists(fig6_path))
print("Fig7 exists:", os.path.exists(fig7_path))
print("Fig8 exists:", os.path.exists(fig8_path))

save_dir = "figures"
save_path = os.path.join(save_dir, "comparison.png")

os.makedirs(save_dir, exist_ok=True)

# -------- load images --------
img6 = Image.open(fig6_path).convert("RGB")
img7 = Image.open(fig7_path).convert("RGB")
img8 = Image.open(fig8_path).convert("RGB")

# -------- resize to same size --------
target_size = (320, 320)
img6 = img6.resize(target_size)
img7 = img7.resize(target_size)
img8 = img8.resize(target_size)

# -------- canvas settings --------
padding = 20
label_height = 40
title_height = 50

canvas_width = target_size[0] * 3 + padding * 4
canvas_height = target_size[1] + title_height + label_height + padding * 2

canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
draw = ImageDraw.Draw(canvas)

# -------- font --------
try:
    font_title = ImageFont.truetype("arial.ttf", 22)
    font_label = ImageFont.truetype("arial.ttf", 18)
except:
    font_title = ImageFont.load_default()
    font_label = ImageFont.load_default()

# -------- title --------
title = "WGAN Reproduction: Figures 6, 7, and 8"
draw.text((canvas_width // 2 - 170, 10), title, fill="black", font=font_title)

# -------- paste images --------
x1 = padding
x2 = x1 + target_size[0] + padding
x3 = x2 + target_size[0] + padding
y = title_height

canvas.paste(img6, (x1, y))
canvas.paste(img7, (x2, y))
canvas.paste(img8, (x3, y))

# -------- labels --------
draw.text((x1 + 90, y + target_size[1] + 8), "Fig 6: DCGAN", fill="black", font=font_label)
draw.text((x2 + 55, y + target_size[1] + 8), "Fig 7: No BN + Const", fill="black", font=font_label)
draw.text((x3 + 75, y + target_size[1] + 8), "Fig 8: MLP Gen", fill="black", font=font_label)

# -------- save --------
canvas.save(save_path)
print(f"Saved comparison image to: {save_path}")

print("Saved to:", os.path.abspath(save_path))