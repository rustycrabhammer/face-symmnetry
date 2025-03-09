import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def analyze_face_frame(frame, face_cascade, eye_cascade):
    # Define regions (same as original)
    regions = {
        'upper_forehead': (0, 0.15),
        'lower_forehead': (0.15, 0.33),
        'upper_eyes': (0.33, 0.39),
        'eyes': (0.39, 0.45),
        'upper_nose': (0.45, 0.55),
        'lower_nose': (0.55, 0.65),
        'upper_lip': (0.65, 0.73),
        'lower_lip': (0.73, 0.80),
        'upper_chin': (0.80, 0.90),
        'lower_chin': (0.90, 1.0)
    }
    
    # Enhanced image processing
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(150, 150),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        return frame, None, None
        
    overlay = frame.copy()
    
    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]
        face_center_x = x + w//2
        
        # Detect eyes for more precise alignment
        eyes = eye_cascade.detectMultiScale(face_region)
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye, right_eye = eyes[:2]
            left_eye_center = (x + left_eye[0] + left_eye[2]//2, y + left_eye[1] + left_eye[3]//2)
            right_eye_center = (x + right_eye[0] + right_eye[2]//2, y + right_eye[1] + right_eye[3]//2)
            face_center_x = (left_eye_center[0] + right_eye_center[0]) // 2
        
        region_scores = {}
        detailed_metrics = {}
        
        # Analyze each region
        for region_name, (start_pct, end_pct) in regions.items():
            start_y = int(y + h * start_pct)
            end_y = int(y + h * end_pct)
            
            pad = 3
            left_half = gray[start_y-pad:end_y+pad, x-pad:face_center_x+pad]
            right_half = gray[start_y-pad:end_y+pad, face_center_x-pad:x+w+pad]
            right_half_flipped = cv2.flip(right_half, 1)
            
            # Ensure same size
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            min_height = min(left_half.shape[0], right_half_flipped.shape[0])
            
            left_half = left_half[:min_height, :min_width]
            right_half_flipped = right_half_flipped[:min_height, :min_width]
            
            # Calculate metrics
            diff = cv2.absdiff(left_half, right_half_flipped)
            mse = np.mean(diff ** 2)
            ssim = 1 - (mse / (255 * 255))
            
            left_edges_x = cv2.Sobel(left_half, cv2.CV_64F, 1, 0)
            right_edges_x = cv2.Sobel(right_half_flipped, cv2.CV_64F, 1, 0)
            left_edges_y = cv2.Sobel(left_half, cv2.CV_64F, 0, 1)
            right_edges_y = cv2.Sobel(right_half_flipped, cv2.CV_64F, 0, 1)
            
            edge_similarity_x = 1 - np.mean(np.abs(left_edges_x - right_edges_x)) / 255
            edge_similarity_y = 1 - np.mean(np.abs(left_edges_y - right_edges_y)) / 255
            
            # Store metrics
            detailed_metrics[region_name] = {
                'structural': ssim,
                'edge_x': edge_similarity_x,
                'edge_y': edge_similarity_y
            }
            
            # Combined score
            region_score = (0.4 * ssim + 0.3 * edge_similarity_x + 0.3 * edge_similarity_y)
            region_scores[region_name] = region_score
            
            # Draw analysis visualization
            cv2.line(overlay, (x, start_y), (x+w, start_y), (255, 255, 255), 1)
            cv2.line(overlay, (face_center_x, start_y), (face_center_x, end_y), (0, 255, 0), 1)
        
        # Draw face rectangle
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 255, 255), 1)
        
        # Calculate overall symmetry
        symmetry_score = np.mean(list(region_scores.values()))
        
        # Blend overlay
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return frame, symmetry_score, detailed_metrics
    
    return frame, None, None

def process_uploaded_image(uploaded_file):
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return frame

def main():
    st.title("Face Symmetry Analysis")
    st.write("Analyze facial symmetry using webcam or upload an image")
    
    # Initialize face and eye cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Add input method selection
    input_method = st.radio("Choose Input Method", ["Webcam", "Upload Image"])
    
    # Create placeholders
    video_placeholder = st.empty()
    metrics_placeholder = st.empty()
    error_placeholder = st.empty()
    
    if input_method == "Upload Image":
        # Image upload section
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            try:
                # Process the uploaded image
                frame = process_uploaded_image(uploaded_file)
                
                # Analyze frame
                processed_frame, symmetry_score, detailed_metrics = analyze_face_frame(frame, face_cascade, eye_cascade)
                
                # Convert BGR to RGB for display
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display the processed image
                video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                
                # Display metrics
                if symmetry_score is not None:
                    metrics_text = f"Overall Symmetry Score: {symmetry_score:.3f}\n\nDetailed Metrics:"
                    if detailed_metrics:
                        for region, metrics in detailed_metrics.items():
                            metrics_text += f"\n\n{region}:"
                            for metric_name, value in metrics.items():
                                metrics_text += f"\n  {metric_name}: {value:.3f}"
                    metrics_placeholder.text(metrics_text)
                else:
                    error_placeholder.error("No face detected in the image. Please try another image.")
                    
            except Exception as e:
                error_placeholder.error(f"Error processing image: {str(e)}")
    
    else:
        # Webcam section
        # Add a start/stop button
        if 'running' not in st.session_state:
            st.session_state.running = False
            
        if 'camera_initialized' not in st.session_state:
            st.session_state.camera_initialized = False
            
        if st.button('Start' if not st.session_state.running else 'Stop'):
            st.session_state.running = not st.session_state.running
            
            # Reset camera initialization when stopping
            if not st.session_state.running:
                st.session_state.camera_initialized = False
                if hasattr(st.session_state, 'cap'):
                    st.session_state.cap.release()
        
        try:
            if st.session_state.running:
                if not st.session_state.camera_initialized:
                    # Try different camera indices
                    for camera_index in [0, 1, 2]:
                        cap = cv2.VideoCapture(camera_index)
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            st.session_state.cap = cap
                            st.session_state.camera_initialized = True
                            error_placeholder.empty()
                            break
                        cap.release()
                    
                    if not st.session_state.camera_initialized:
                        error_placeholder.error("""
                            Unable to access webcam. Please check:
                            1. Your webcam is properly connected
                            2. You've granted browser permission to access the webcam
                            3. No other application is using the webcam
                            4. Try refreshing the page
                        """)
                        st.session_state.running = False
                        return
                
                # Capture and process frame
                ret, frame = st.session_state.cap.read()
                if not ret:
                    error_placeholder.error("Failed to grab frame from camera")
                    st.session_state.running = False
                    return
                
                frame = cv2.flip(frame, 1)
                
                # Analyze frame
                processed_frame, symmetry_score, detailed_metrics = analyze_face_frame(frame, face_cascade, eye_cascade)
                
                # Convert BGR to RGB for display
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Update the video feed
                video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                
                # Display metrics
                if symmetry_score is not None:
                    metrics_text = f"Overall Symmetry Score: {symmetry_score:.3f}\n\nDetailed Metrics:"
                    if detailed_metrics:
                        for region, metrics in detailed_metrics.items():
                            metrics_text += f"\n\n{region}:"
                            for metric_name, value in metrics.items():
                                metrics_text += f"\n  {metric_name}: {value:.3f}"
                    metrics_placeholder.text(metrics_text)
        
        except Exception as e:
            error_placeholder.error(f"Error: {str(e)}")
            st.session_state.running = False
            if hasattr(st.session_state, 'cap'):
                st.session_state.cap.release()
        
        # Clean up when stopping
        if not st.session_state.running and hasattr(st.session_state, 'cap'):
            st.session_state.cap.release()

if __name__ == "__main__":
    main()
