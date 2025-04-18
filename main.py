import cv2
import mediapipe as mp
import time

class BJJScoreKeeper:
    def __init__(self):
        self.player1_score = 0
        self.player2_score = 0
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

    def detect_gesture(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, 
                                         self.mp_hands.HAND_CONNECTIONS)
                # Here you would add gesture recognition logic
                # Example: detect if fingers are raised in specific patterns
                
    def run(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
                
            self.detect_gesture(frame)
            
            # Display scores
            cv2.putText(frame, f"Player 1: {self.player1_score}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Player 2: {self.player2_score}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('BJJ Score Keeper', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    scorekeeper = BJJScoreKeeper()
    scorekeeper.run()