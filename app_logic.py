import threading
import time
from helper.helper import WebcamVideoStream
import imutils
import cv2
from helper.helper import FPS


def show_text_on_frame(image,string,pos):
	cv2.putText(image,string,(pos),cv2.FONT_HERSHEY_PLAIN, 0.75, (77, 255, 9), 2)

class MyThread(threading.Thread):
	def __init__(self):
		self.isRunning = True

	def segment(self):
		print("Segment started!")              
		#logic
		time.sleep(1)                                      
		print("Segment finished!")            

	def display(self):
		stream = WebcamVideoStream(src=0).start()
		fps = FPS(5).start()
		frame = stream.read()
		key = cv2.waitKey(1) or 0xff
		fps.update()
		show_text_on_frame(frame,"fps: {}".format(int(fps.fps_local())),(10,30))
		# keybindings for display
		# if key == ord('p'):  # pause
		# 	while True:
		# 		key2 = cv2.waitKey(1) or 0xff
		# 		if key2 == ord('p'):  # resume
		# 			break	
		cv2.imshow('frame', frame)
		if key == (ord('q')):  # exit
			fps.stop()
			cv2.destroyAllWindows()
			stream.stop()

