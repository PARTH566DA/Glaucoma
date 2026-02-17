import cv2
import os
import sys

# Example paths (update the base path to your download location)
base_path = "/Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135/"
device = "Remidio"  # Options: "Bosch", "Forus", "Remidio"
dataset_split = "Train"  # Options: "Train", "Test"
image_filename = "IMG_3710.jpg"

# Allow overriding via command-line arguments: python test.py <image_filename> [device] [dataset_split]
if len(sys.argv) > 1:
	image_filename = sys.argv[1]
if len(sys.argv) > 2:
	device = sys.argv[2]
if len(sys.argv) > 3:
	dataset_split = sys.argv[3]

print(f"Loading from: dataset={dataset_split}, device={device}, image={image_filename}")

def find_case_insensitive_file(directory, filename):
	"""Try to find a file in directory matching filename case-insensitively."""
	try:
		for f in os.listdir(directory):
			if f.lower() == filename.lower():
				return os.path.join(directory, f)
	except Exception:
		return None
	return None

# Load original fundus image with checks
fundus_dir = os.path.join(base_path, dataset_split, '1.0_Original_Fundus_Images', device)
img_path = os.path.join(fundus_dir, image_filename)
if not os.path.isfile(img_path):
	alt = None
	if os.path.isdir(fundus_dir):
		alt = find_case_insensitive_file(fundus_dir, image_filename)
		if alt:
			print(f"Found case-insensitive match for image: {alt}")
			img_path = alt
		else:
			print(f"Image file not found at: {img_path}")
			contents = os.listdir(fundus_dir)
			print(f"Contents of device directory ({fundus_dir}) (showing up to 20 entries):\n", contents[:20])
			sys.exit(1)
	else:
		print(f"Device directory does not exist: {fundus_dir}")
		sys.exit(1)

fundus_image = cv2.imread(img_path)
if fundus_image is None:
	print(f"cv2.imread returned None for: {img_path} (file may be corrupted or unreadable)")
	sys.exit(1)
fundus_image_rgb = cv2.cvtColor(fundus_image, cv2.COLOR_BGR2RGB)

# Load separate disc and cup masks from STAPLE directory
mask_basename = os.path.splitext(image_filename)[0]
disc_dir = os.path.join(base_path, dataset_split, '5.0_OD_OC_Mean_Median_Majority_STAPLE', device, 'Disc', 'STAPLE')
cup_dir = os.path.join(base_path, dataset_split, '5.0_OD_OC_Mean_Median_Majority_STAPLE', device, 'Cup', 'STAPLE')

# Try to load disc mask
optic_disc_mask = None
disc_mask_path = None
if os.path.isdir(disc_dir):
	# Try with .png extension first (most common for masks)
	for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
		p = os.path.join(disc_dir, mask_basename + ext)
		if os.path.isfile(p):
			disc_mask_path = p
			break
	# Fallback: case-insensitive search
	if disc_mask_path is None:
		for f in os.listdir(disc_dir):
			if f.lower().startswith(mask_basename.lower()):
				disc_mask_path = os.path.join(disc_dir, f)
				break
	
	if disc_mask_path:
		optic_disc_mask = cv2.imread(disc_mask_path, cv2.IMREAD_GRAYSCALE)
		if optic_disc_mask is None:
			print(f"Warning: cv2.imread returned None for disc mask: {disc_mask_path}")
else:
	print(f"Warning: Disc directory does not exist: {disc_dir}")

# Try to load cup mask
optic_cup_mask = None
cup_mask_path = None
if os.path.isdir(cup_dir):
	# Try with .png extension first (most common for masks)
	for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
		p = os.path.join(cup_dir, mask_basename + ext)
		if os.path.isfile(p):
			cup_mask_path = p
			break
	# Fallback: case-insensitive search
	if cup_mask_path is None:
		for f in os.listdir(cup_dir):
			if f.lower().startswith(mask_basename.lower()):
				cup_mask_path = os.path.join(cup_dir, f)
				break
	
	if cup_mask_path:
		optic_cup_mask = cv2.imread(cup_mask_path, cv2.IMREAD_GRAYSCALE)
		if optic_cup_mask is None:
			print(f"Warning: cv2.imread returned None for cup mask: {cup_mask_path}")
else:
	print(f"Warning: Cup directory does not exist: {cup_dir}")

# Print status
if optic_disc_mask is not None and optic_cup_mask is not None:
	print(f"Loaded fundus image: {img_path} (shape={fundus_image.shape})")
	print(f"Loaded disc mask: {disc_mask_path} (shape={optic_disc_mask.shape})")
	print(f"Loaded cup mask: {cup_mask_path} (shape={optic_cup_mask.shape})")
elif optic_disc_mask is not None:
	print(f"Loaded fundus image: {img_path} (shape={fundus_image.shape})")
	print(f"Loaded disc mask: {disc_mask_path} (shape={optic_disc_mask.shape})")
	print(f"Warning: No cup mask found for '{mask_basename}'")
elif optic_cup_mask is not None:
	print(f"Loaded fundus image: {img_path} (shape={fundus_image.shape})")
	print(f"Loaded cup mask: {cup_mask_path} (shape={optic_cup_mask.shape})")
	print(f"Warning: No disc mask found for '{mask_basename}'")
else:
	print(f"Loaded fundus image: {img_path} (shape={fundus_image.shape})")
	print(f"Warning: No masks found for '{mask_basename}' in device '{device}'")