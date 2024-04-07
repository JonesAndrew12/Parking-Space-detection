from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('lists/best.pt')

# Perform detection on the video source
result = model(source='lists/parking.mp4', show=True, save=True)

# Process the result
if isinstance(result, list):
    # Iterate over each frame in the result list
    for frame_result in result:
        # Extract bounding boxes from the 'boxes' attribute
        boxes = frame_result.boxes

        # Filter frames for both 'empty' and 'occupied'
        target_boxes = [box for box in boxes if box['name'] in ['empty', 'occupied']]

        # Print the target bounding boxes
        for box in target_boxes:
            print(box)
else:
    # If the result is not a list, handle the error or unexpected result accordingly
    print("Unexpected result format. Check the source or model configuration.")