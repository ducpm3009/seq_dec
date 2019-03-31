import os
import time
from imutils.video import WebcamVideoStream
from helper.helper import FPS
import imutils
import cv2


def show_text_on_frame(image,string,pos):
	cv2.putText(image,string,(pos),cv2.FONT_HERSHEY_PLAIN, 0.75, (77, 255, 9), 2)

def segment():
	print("Segment started!")
	#logic
	time.sleep(1)
	print("Segment finished!")

def human_dec(frame):
	person_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
	rects = person_cascade.detectMultiScale(gray_frame)
	return rects

def display():
	vs = WebcamVideoStream(src=0).start()
	fps = FPS(5).start()
	while True:
		frame = vs.read()
		frame = imutils.resize(frame, width=360, height = 640)
		key = cv2.waitKey(1) & 0xFF
		body_rec = []
		fps.update()
		show_text_on_frame(frame,"{}".format(int(fps.fps_local())),(10,30))
		rects = human_dec(frame)
		for (x, y, w, h) in rects:
			body_rec.append(cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2))
			print(*body_rec, sep = '/n')
			#logic 


		cv2.imshow("Frame", frame)
		if key == ord('p'):  # pause
			while True:
				key2 = cv2.waitKey(1) or 0xff
				if key2 == ord('p'):  # resume
					break

		if key == (ord('q')):  # exit
			fps.stop()
			cv2.destroyAllWindows()
			stream.stop()

