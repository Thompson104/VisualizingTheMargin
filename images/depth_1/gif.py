import imageio
from natsort import natsorted
import os 

images = []
for f in natsorted(os.listdir()):
	if ".jpg" in f: images.append(imageio.imread(str(f)))

print(images)

imageio.mimsave("animation.gif", images, duration=.2)
