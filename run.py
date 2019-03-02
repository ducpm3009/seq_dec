# import the necessary packages
from __future__ import print_function
from imutils.video import WebcamVideoStream
import argparse
import imutils
import cv2
from helper.helper import FPS

def show_fps(image,string,pos):
    cv2.putText(image,string,(pos),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

stream = WebcamVideoStream(src=0).start()
fps = FPS(5).start()

while True:
	# grab next frame
	frame = stream.read()
	key = cv2.waitKey(1) or 0xff
	fps.update()
	# keybindings for display
	show_fps(frame,"fps: {}".format(int(fps.fps_local())),(10,30))
	if key == ord('p'):  # pause
		while True:
			key2 = cv2.waitKey(1) or 0xff

			#MAIN LOGIC :
			#TODO : use threading to improve

			if key2 == ord('p'):
				# resume
				break	
	cv2.imshow('frame', frame)
	if key == (ord('q')):  # exit
		fps.stop()
		cv2.destroyAllWindows()
		stream.stop()	
		break 
