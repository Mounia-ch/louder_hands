import cv2
import mediapipe as mp
import time
import os

class handTracker():
    def __init__(self, letter, *, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5, samples=100):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.letter = letter
        self.samples = samples

    def handsFinder(self,image,draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self,image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])
            if draw:
                cv2.circle(image,(cx,cy), 5 , (255,0,255), cv2.FILLED)

        return lmlist

def main():

    letter = 24
    samples = 25
    cap = cv2.VideoCapture(0)
    tracker = handTracker(letter, samples=samples)

    path = '/Users/manuel/Pictures/'
    if not os.path.isdir(path+'draw/'):
        os.mkdir(path+'draw/')
    if not os.path.isdir(path+'raw/'):
        os.mkdir(path+'raw/')
    
    if not os.path.isdir(path+'draw/'+str(letter)+'/'):
        os.mkdir(path+'draw/'+str(letter)+'/')
    
    if not os.path.isdir(path+'raw/'+str(letter)+'/'):
        os.mkdir(path+'raw/'+str(letter)+'/')
    if not os.path.isfile(path+'letters.txt'):
        with open(path+'letters.txt', 'w') as f:
            pass

    pictures = os.listdir(path+'draw/'+str(letter)+'/')
    if '.DS_Store' in pictures:
        pictures.remove('.DS_Store')
    if not len(pictures)==0:
        pictures = [int(picture.replace('.png', '')) for picture in pictures]
        i=max(pictures)+1
        maxi = i+tracker.samples
    else:
        i=0
        maxi = tracker.samples
    while i<=maxi:
        success,image = cap.read()
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)
        if len(lmList) != 0:
            max_width = 0
            min_width = 1000000
            max_height = 0
            min_height = 1000000
            for point in lmList:
                if point[1]>max_width:
                    max_width=point[1]
                if point[1]<min_width:
                    min_width=point[1]
                if point[2]>max_height:
                    max_height=point[2]
                if point[2]<min_height:
                    min_height=point[2]
            min_width = min_width-50
            max_width = max_width+50
            min_height = min_height-50
            max_height = max_height+50         
            cropped_image = image[min_height:max_height, min_width:max_width]
            cv2.imwrite(path + 'draw/'+ str(tracker.letter) + '/'+str(i)+'.png', cropped_image)
            success,raw_image = cap.read()
            raw_image = tracker.handsFinder(raw_image, draw=False)
            raw_cropped_image = raw_image[min_height:max_height, min_width:max_width]
            cv2.imwrite(path + 'raw/'+ str(tracker.letter) + '/'+str(i)+'.png', raw_cropped_image)
            with open(path+'letters.txt', "a") as f:
                lmList_char = [str(element) for element in lmList]
                f.write('\t'.join(lmList_char) + '\t' + str(letter) + '\n')
            print(i)
            i+=1
            time.sleep(0.2)

        cv2.imshow("Video",image)
        
        cv2.waitKey(1)

if __name__ == "__main__":
    main()