import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo
import ros_numpy
import os
import sys
import atexit
import termios
import select

class ARMarkerDetection:
    def __init__(self):
        # Set the dictionary and parameters for marker detection
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
        self.parameters = cv2.aruco.DetectorParameters()

        # Initialize the KBHit object for keyboard input
        self.kb = self.KBHit()

        print("Press 's' to save the marker corners. Press 'q' to quit.")

        # Subscribe to the image topic
        self.image_subscriber = rospy.Subscriber('/camera/color/image_rect_color', Image, self.image_callback)
        self.camera_info_subscriber = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)

        # Variables for storing marker information
        self.marker_ids = []
        self.marker_pose = []

    # Define the KBHit class for keyboard input handling
    class KBHit:
        def __init__(self):
            if os.name == 'nt':
                pass
            else:
                # Save the terminal settings
                self.fd = sys.stdin.fileno()
                self.new_term = termios.tcgetattr(self.fd)
                self.old_term = termios.tcgetattr(self.fd)
                # New terminal setting unbuffered
                self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
                termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)
                # Support normal-terminal reset at exit
                atexit.register(self.set_normal_term)

        def set_normal_term(self):
            if os.name == 'nt':
                pass
            else:
                termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)

        def getch(self):
            s = ''
            if os.name == 'nt':
                return msvcrt.getch().decode('utf-8')
            else:
                return sys.stdin.read(1)

        def kbhit(self):
            if os.name == 'nt':
                return msvcrt.kbhit()
            else:
                dr, dw, de = select.select([sys.stdin], [], [], 0)
                return dr != []

    # Function to save marker corners
    def save_marker_info(self, output_path):

        np.save(output_path+"marker_ids", self.marker_ids)
        np.save(output_path+"marker_pose", self.marker_pose)

        print(f"Marker corners saved as {output_path}")

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.K).reshape((3, 3))
        self.distortion_coeffs = np.array(msg.D)

        # Unsubscribe from the camera info topic after receiving the parameters
        self.camera_info_subscriber.unregister()

    # ROS callback function for image subscriber
    def image_callback(self, msg):
        # Convert the ROS image message to a NumPy array
        frame = ros_numpy.numpify(msg)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers in the frame
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.0538, self.camera_matrix, self.distortion_coeffs)

        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)[:,:,(2,1,0)]

        # Sort the corners and IDs based on the marker IDs
        if ids is not None and len(ids) > 0:
            indices = np.argsort(ids.flatten())
            sorted_ids = []
            sorted_pose = []

            for index in indices:
                sorted_ids.append(ids[index])

                camera_pose = np.eye(4)
                camera_pose[:3, :3] = cv2.Rodrigues(rvec[index])[0]
                camera_pose[:3, 3] = tvec[index]

                sorted_pose.append(camera_pose)

            self.marker_ids = sorted_ids
            self.marker_pose = sorted_pose

        # Display the frame
        cv2.imshow('AR Marker Detection', frame)

        # Check if 's' key is pressed
        if self.kb.kbhit() and self.kb.getch() == 's':
            # Save the marker corners to a file
            self.save_marker_info(output_path='/home/kist/workspace/playground/armarker/')
            print("Marker corners saved. Press 's' again to save another.")

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown('Quit')

    def run(self):
        # Initialize the ROS node
        rospy.init_node('ar_marker_detection')

        # Enter the ROS event loop
        rospy.spin()

if __name__ == '__main__':
    # Create an instance of the ARMarkerDetection class
    marker_detection = ARMarkerDetection()
    marker_detection.run()
