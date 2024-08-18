import os
import cv2
import time
import base64
import requests
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.uix.anchorlayout import AnchorLayout

class ObjectDetectionApp(App):
    def build(self):
        # Layout containing images and buttons
        layout = BoxLayout(orientation='vertical')

        # Image display widgets
        self.img = Image(size_hint=(1, 0.8))
        self.processed_img = Image(size_hint=(1, 0.8))

        # Start and Stop buttons
        self.start_button = Button(text='Start', size_hint=(1, 0.1))
        self.start_button.bind(on_press=self.start_detection)

        self.stop_button = Button(text='Stop', size_hint=(1, 0.1))
        self.stop_button.bind(on_press=self.stop_detection)

        # Add image and button widgets to layout
        layout.add_widget(self.img)
        layout.add_widget(self.processed_img)
        layout.add_widget(self.start_button)
        layout.add_widget(self.stop_button)

        # Initialize video capture and variables
        self.capture = cv2.VideoCapture(0)
        self.api_url = 'http://103.82.134.238:7123/predict'
        self.detecting = False
        self.image_count = 0
        self.frame_count = 0
        self.start_time = time.time()

        # Create directory for saving images
        # if not os.path.exists('image'):
        #     os.makedirs('image')

        return layout

    def start_detection(self, instance):
        if not self.detecting:
            self.detecting = True
            self.start_button.text = 'Detecting...'
            Clock.schedule_interval(self.update, 1.0 / 30.0) # 30 FPS
            Clock.schedule_interval(self.print_fps, 1.0) # Print FPS reality every second

    def stop_detection(self, instance):
        if self.detecting:
            self.detecting = False
            self.start_button.text = 'Start'
            Clock.unschedule(self.update)
            Clock.unschedule(self.print_fps)

    def update(self, dt):
        if not self.detecting:
            return
        
        ret, frame = self.capture.read()
        if ret:
            self.frame_count += 1
            frame_b64 = self.convert_to_base64(frame)
            results = self.predict_image(frame_b64)
            if results:
                self.draw_results(frame, results)
                #self.save_image(frame)
                buf = cv2.flip(frame, 0).tobytes()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.img.texture = texture

    def print_fps(self, dt):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        fps = self.frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        self.frame_count = 0
        self.start_time = current_time

    def convert_to_base64(self, img):
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')

    def predict_image(self, base64_image):
        payload = {"base64_image": base64_image}
        response = requests.post(self.api_url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Request failed with status code {response.status_code}")
            return None

    def draw_results(self, frame, results):
        if isinstance(results, dict):
            result_data = results.get('result', {})
            bounding_box = results.get('bounding_box', {})

            age = result_data.get('age', {}).get('0', '')
            race = result_data.get('race', {}).get('0', '')
            masked = result_data.get('masked', {}).get('0', '')
            skintone = result_data.get('skintone', {}).get('0', '')
            emotion = result_data.get('emotion', {}).get('0', '')
            gender = result_data.get('gender', {}).get('0', '')

            labels = [
                f"Age: {age}",
                f"Race: {race}",
                f"Masked: {masked}",
                f"Skintone: {skintone}",
                f"Emotion: {emotion}",
                f"Gender: {gender}"
            ]

            ih, iw = frame.shape[:2]
            x_bb = int(bounding_box.get('x', 0) * iw)
            y_bb = int(bounding_box.get('y', 0) * ih)

            # Draw bounding box
            self.draw_bounding_box(frame, bounding_box, ih, iw)

            # Draw labels
            for i, label in enumerate(labels):
                y_position = y_bb - 10 - (i * 25)  # Adjust vertical position for each line
                cv2.putText(frame, label, (x_bb, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    def draw_bounding_box(self, frame, bounding_box, ih, iw):
        if bounding_box:
            x_bb = int(bounding_box.get('x', 0) * iw)
            y_bb = int(bounding_box.get('y', 0) * ih)
            w_bb = int(bounding_box.get('width', 0) * iw)
            h_bb = int(bounding_box.get('height', 0) * ih)
            cv2.rectangle(frame, (x_bb, y_bb), (x_bb + w_bb, y_bb + h_bb), (0, 255, 0), 2)
            face_region = frame[y_bb:y_bb + h_bb, x_bb:x_bb + w_bb]
            return face_region
        return None

    # def save_image(self, frame):
    #     self.image_count += 1
    #     image_path = os.path.join('image', f'processed_image_{self.image_count}.jpg')
    #     cv2.imwrite(image_path, frame)

if __name__ == '__main__':
    ObjectDetectionApp().run()
