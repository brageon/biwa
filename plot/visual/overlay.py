from PIL import Image
def overlay_pictures(picture1, picture2):
  if picture1.size[0] < picture2.size[0] or picture1.size[1] < picture2.size[1]:
    picture1 = picture1.resize((max(picture1.size[0], picture2.size[0]), max(picture1.size[1], picture2.size[1])), Image.ANTIALIAS)
  elif picture2.size[0] < picture1.size[0] or picture2.size[1] < picture1.size[1]:
    picture2 = picture2.resize((max(picture2.size[0], picture1.size[0]), max(picture2.size[1], picture1.size[1])), Image.ANTIALIAS)
  output_picture = Image.new("RGB", (picture1.size[0], picture1.size[1]), "white")
  output_picture.paste(picture1, (0, 0))
  transparency_mask = picture2.copy()
  transparency_mask = transparency_mask.convert("L")  # Convert to grayscale
  transparency_mask = transparency_mask.point(lambda x: x * 0.5)  # Apply a transparency effect
  output_picture.paste(picture2, (0, 0), mask=transparency_mask)
  output_picture.save("imp.png")
if __name__ == "__main__":
  picture1 = Image.open("inp_1.png")
  picture2 = Image.open("inp_2.png")
  overlay_pictures(picture1, picture2)
