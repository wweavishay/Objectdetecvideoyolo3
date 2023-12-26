import cv2
import numpy as np
import os
import time
import calcdistance
import calcdistancemathway


class ObjectDetection:
    def __init__(self, config_path, weights_path, classes_path, output_folder):
        self.config_path = config_path
        self.weights_path = weights_path
        self.classes_path = classes_path
        self.output_folder = output_folder
        self.draw_number = 0  # Initialize a counter for the draw number ID

        # Load YOLO model
        self.net = cv2.dnn.readNet(weights_path, config_path)

        # Load classes
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Generate random colors for each class
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # Initialize previous positions
        self.prev_positions = {}

        # Initialize detected objects IDs
        self.detected_objects_ids = set()

        # Initialize routes
        self.routes = {}

        # Dictionary to store object counts
        self.object_counts = {class_name: 0 for class_name in self.classes}

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers

    def detect_objects(self, image):
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)

        # Get detections from YOLO model
        outs = self.net.forward(self.get_output_layers())

        class_ids = []
        confidences = []
        self.boxes = []
        conf_threshold = 0.6
        nms_threshold = 0.4

        # Process the detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    self.boxes.append([x, y, w, h])

        # Apply non-maximum suppression to remove redundant detections
        indices = cv2.dnn.NMSBoxes(self.boxes, confidences, conf_threshold, nms_threshold)

        # Process the detections and update speed
        for i in indices:
            try:
                self.box = self.boxes[i]
            except:
                i = i[0]
                self.box = self.boxes[i]

            x = self.box[0]
            y = self.box[1]
            w = self.box[2]
            h = self.box[3]
            class_name = self.classes[class_ids[i]]
            obj_id = f"{class_name}_{i}"
            if obj_id in self.detected_objects_ids:
                continue  # Skip counting this object
            else:
                # Increment the count for the detected object
                self.object_counts[class_name] += 1
                # Add the object ID to the set of detected objects
                self.detected_objects_ids.add(obj_id)

            # Calculate the current position of the detected object
            center_x = x + w // 2
            center_y = y + h // 2

            # Record the route of the object
            if class_name not in self.routes:
                self.routes[class_name] = []
            self.routes[class_name].append((center_x, center_y))

            # Get the previous position and timestamp for the current object
            prev_position = self.prev_positions.get(class_name)

            if prev_position is not None:
                # Calculate the speed using the Euclidean distance and time elapsed
                prev_x, prev_y, prev_timestamp = prev_position
                distance = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                time_elapsed = 1  # Replace with the actual time elapsed between frames
                speed = distance / time_elapsed
            else:
                # No previous position available, set speed to None
                speed = None

            # Update the previous position and timestamp for the current object
            self.prev_positions[class_name] = (center_x, center_y, time.time())

            # Draw the bounding box with confidence and speed labels
            self.draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w),
                                     round(y + h), speed)

    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h, speed):
        fps = 30
        class_name = self.classes[class_id]

        if speed is not None:
            # Convert speed from px/frame to km/h using the hypothetical conversion factor and frame rate
            conversion_factor = 0.1  # 1 pixel corresponds to 0.1 meters (adjust as needed)
            speed_km_per_h = speed * conversion_factor * fps * 3.6 / 1000  # Convert to km/h (1 m/s = 3.6 km/h)
        else:
            speed_km_per_h = None

        speed_label = f"{speed_km_per_h:.2f} km/h" if speed_km_per_h is not None else "Speed: N/A"
        confidence_label = f"{class_name}:{confidence:.2f}({speed_label}) "
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1

        # Increment the draw number ID for each object
        self.draw_number += 1

        # Add the draw number ID to the label
        draw_number_label = f"ID: {self.draw_number}"
        label = confidence_label + draw_number_label

        # Draw the bounding box
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

        # Determine the text color based on the brightness of the bounding box color
        brightness = sum(color) // 3  # Simple average of RGB values
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)  # Use white or black text

        # Draw the confidence label as background
        label_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(img, (x, y - 2 * label_size[1]), (x + label_size[0], y), color, cv2.FILLED)

        # Draw the confidence label text
        cv2.putText(img, confidence_label, (x, y - 5), font, font_scale, text_color, font_thickness)

        # Draw the draw number ID
        draw_number_size, _ = cv2.getTextSize(draw_number_label, font, font_scale, font_thickness)
        cv2.putText(img, draw_number_label, (x, y + 5 + draw_number_size[1]), font, font_scale, text_color, font_thickness)


    def save_detected_objects(self,image, boxes, class_ids, classes, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for i, box in enumerate(boxes):
            class_id = class_ids[i]
            class_name = classes[class_id]
            x, y, w, h = box

            # Perform error-checking on bounding boxes
            if w <= 0 or h <= 0:
                print(f"Skipping saving {class_name}_{i}.jpg due to an invalid bounding box.")
                continue

            object_image = image[y:y + h, x:x + w]

            # Check if object_image is empty or None
            if object_image is None or object_image.size == 0:
                print(f"Skipping saving {class_name}_{i}.jpg due to an empty object image.")
                continue

            # Create a folder for the class if it doesn't exist
            class_output_folder = os.path.join(output_folder, class_name)
            if not os.path.exists(class_output_folder):
                os.makedirs(class_output_folder)

            # Get the next available index for the filename
            idx = len(os.listdir(class_output_folder))
            output_path = os.path.join(class_output_folder, f"{class_name}_{idx}.jpg")
            cv2.imwrite(output_path, object_image)


    def draw_object_counters(self,image, object_counts, classes, screen_height, dictmaxcount):
        side_menu_width = 170
        side_menu_color = (173, 216, 230)
        image_height, image_width = image.shape[:2]

        # Rectangle coordinates to position the panel in the left-up corner of the image
        panel_left_x = 0
        panel_right_x = side_menu_width
        panel_top_y = 0
        panel_bottom_y = min(280, image_height)  # Limit the height of the panel to a maximum of 280 pixels

        # Create a half-transparent panel
        overlay = image.copy()
        cv2.rectangle(overlay, (panel_left_x, panel_top_y), (panel_right_x, panel_bottom_y), side_menu_color, -1)
        alpha = 0.5  # Adjust alpha for desired transparency level
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_color = (0, 0, 0)  # Black color for the text

        text_y_offset = 40
        valid_counts = {class_name: count for class_name, count in object_counts.items() if count > 0}

        # Keep track of objects already counted
        counted_objects = set()

        for idx, (class_name, count) in enumerate(valid_counts.items()):
            # Check if the object has already been counted
            if class_name in counted_objects:
                continue

            maxcount = 0
            for key, value in dictmaxcount.items():
                if key == class_name:
                    maxcount = value

            text = f"{class_name}: {maxcount}"  # Use count instead of textcount
            # Position the text on the left-up panel
            text_position = (panel_left_x + 30, panel_top_y + (idx + 1) * text_y_offset)

            cv2.putText(image, text, text_position, font, font_scale, text_color, font_thickness)

            # Add the object to the counted_objects set
            counted_objects.add(class_name)


    def printroute(self, image, object_counts, classes, screen_height, routes):
        print("Routes:")
        self.class_max_dict = {}

        for class_name, route in routes.items():
            route_str = " -> ".join([f"({x},{y})" for x, y in route])
            print(f"Class '{class_name}': ")
            # calcdistance
            num_vehicles, vehicle_routes = calcdistance.analyze_vehicle_routes(route)
            vehicle_routes_str = calcdistance.get_vehicle_routes(vehicle_routes)


            for vehicle_id, route_str in vehicle_routes_str.items():
                print(f"{class_name} {vehicle_id + 1} route: {route_str}")
                max = vehicle_id + 1
                self.class_max_dict[class_name] = max

            # Draw object counters after analyzing vehicle routes
            self.draw_object_counters(image, object_counts, classes, screen_height, self.class_max_dict)

            print("---------------------")

    def save_unique_routes_to_file(self, image, object_counts, classes, screen_height, routes, output_file):
        unique_routes = []

        with open(output_file, 'w') as file:
            for class_name, route in routes.items():
                route_str = " -> ".join([f"({x},{y})" for x, y in route])
                print(f"Class '{class_name}': ")
                num_vehicles, vehicle_routes = calcdistance.analyze_vehicle_routes(route)
                vehicle_routes_str = calcdistance.get_vehicle_routes(vehicle_routes)

                class_title = f"Class '{class_name}' routes:"
                file.write(class_title + '\n')
                print(class_title)

                for vehicle_id, route_str in vehicle_routes_str.items():
                    print(f"{class_name} {vehicle_id + 1} route: {route_str}")

                    # Check if the class_name and route_str tuple is already in unique_routes
                    route_already_saved = any((class_name, route_str) in saved_route for saved_route in unique_routes)

                    if not route_already_saved:
                        # Add the new class_name and route_str to the list if it's unique
                        unique_routes.append((class_name, route_str))
                    else:
                        # Replace the existing class_name and route_str if the new route is longer
                        for idx, saved_route in enumerate(unique_routes):
                            saved_class_name, saved_route_str = saved_route
                            if (class_name, route_str) == (saved_class_name, saved_route_str) and len(route_str) > len(
                                    saved_route_str):
                                unique_routes[idx] = (class_name, route_str)

                    # Write the route to the file
                    file.write(f"{class_name} {vehicle_id + 1} route: {route_str}\n")

                # Add a blank line to separate classes in the file
                file.write('\n')
                print("---------------------")



    def run(self, video_path):
        # Initialize variables
        detected_objects_ids = set()
        routes = {}
        self.class_max_dict = {}  # Initialize class_max_dict here

        # Load video and classes
        cap = cv2.VideoCapture(video_path)
        with open(self.classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        # Generate random colors for each class
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        # Load YOLO model
        net = cv2.dnn.readNet(self.weights_path, self.config_path)

        # Get the screen size
        screen_width, screen_height = 1100, 600  # Replace with your screen resolution

        # Dictionary to store object counts
        object_counts = {class_name: 0 for class_name in classes}

        # Main loop for processing video frames
        while True:
            ret, image = cap.read()
            if not ret:
                break

            # Calculate the aspect ratio between the original image and the screen size
            aspect_ratio = screen_width / float(image.shape[1])
            new_height = int(image.shape[0] * aspect_ratio)

            # Resize the image to fit the screen size
            image = cv2.resize(image, (screen_width, new_height))

            # Draw the object counters
            self.draw_object_counters(image, object_counts, classes, screen_height, self.class_max_dict)




            # Preprocess the image for YOLO model
            Width = image.shape[1]
            Height = image.shape[0]
            scale = 0.00392
            blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)

            # Get detections from YOLO model
            outs = net.forward(self.get_output_layers())

            class_ids = []
            confidences = []
            self.boxes = []
            conf_threshold = 0.6
            nms_threshold = 0.4

            # Process the detections
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > conf_threshold:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w // 2
                        y = center_y - h // 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        self.boxes.append([x, y, w, h])

            # Apply non-maximum suppression to remove redundant detections
            indices = cv2.dnn.NMSBoxes(self.boxes, confidences, conf_threshold, nms_threshold)




            # Process the detections and update speed
            for i in indices:
                try:
                    self.box = self.boxes[i]
                except:
                    i = i[0]
                    self.box = boxes[i]

                x = self.box[0]
                y = self.box[1]
                w = self.box[2]
                h = self.box[3]
                class_name = classes[class_ids[i]]
                obj_id = f"{class_name}_{i}"
                if obj_id in detected_objects_ids:
                    continue  # Skip counting this object
                else:
                    # Increment the count for the detected object
                    object_counts[class_name] += 1
                    # Add the object ID to the set of detected objects
                    detected_objects_ids.add(obj_id)

                # Calculate the current position of the detected object
                center_x = x + w // 2
                center_y = y + h // 2

                # Record the route of the object
                if class_name not in routes:
                    routes[class_name] = []
                routes[class_name].append((center_x, center_y))

                # Get the previous position and timestamp for the current object
                prev_position = self.prev_positions.get(class_name)

                if prev_position is not None:
                    # Calculate the speed using the Euclidean distance and time elapsed
                    prev_x, prev_y, prev_timestamp = prev_position
                    distance = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                    time_elapsed = 1  # Replace with the actual time elapsed between frames
                    speed = distance / time_elapsed
                else:
                    # No previous position available, set speed to None
                    speed = None

                # Update the previous position and timestamp for the current object
                self.prev_positions[class_name] = (center_x, center_y, time.time())

                # Draw the bounding box with confidence and speed labels
                self.draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w),
                                     round(y + h), speed)

                self.save_detected_objects(image, self.boxes, class_ids, classes, output_folder)

            # Show the route of the object
            self.printroute(image, object_counts, classes, screen_height, routes)

            # Draw the object counters
            self.draw_object_counters(image, object_counts, classes, screen_height, self.class_max_dict)

            # Display the processed image
            cv2.imshow("Object Detection", image)

            # Check for user key press to exit the loop
            key = cv2.waitKey(1)
            if key == 27:  # Press 'ESC' to exit
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        # After the loop ends, save the unique routes to a file
        output_file = "unique_routes.txt"
        self.save_unique_routes_to_file(image, object_counts, classes, screen_height, routes, output_file)



if __name__ == "__main__":
    video_path = 'video/vid4.mp4'  # Replace with the path to your video file
    config_path = 'yolov3.cfg'
    weights_path = 'yolov3.weights'
    classes_path = 'yolov3.txt'
    output_folder = 'detected_objects'  # Replace with the desired output folder

    obj_detection = ObjectDetection(config_path, weights_path, classes_path, output_folder)
    obj_detection.run(video_path)
