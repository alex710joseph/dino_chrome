#import keyboard
import numpy as np 
from PIL import ImageGrab
import cv2 
import time
from pynput.keyboard import Key,Controller
import tensorflow as tf

datadirs="D:/flydinofly/success/"
datadirf="D:/flydinofly/failure/"

kb=Controller()

model=tf.keras.models.load_model("dinasour6.model")


def roi(img,vertices):
	mask=np.zeros_like(img)
	cv2.fillPoly(mask,vertices,255)
	masked=cv2.bitwise_and(img,mask)
	return masked
	

def process_img(unprocessed):

	processed_img=cv2.Canny(unprocessed,threshold1=200,threshold2=300)
	vertices=np.array([[220,70],[220,250], [320,250], [320,70]], np.int32)
	processed_img=roi(processed_img,[vertices])
	return processed_img



def main():
	i=0
	j=0
	set=0
	last=time.time()
	while(True):
		flag=1
		screen=np.array(ImageGrab.grab(bbox=(25,100,500,400)))
		screen=cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
		new_screen=process_img(screen)
		cv2.imshow('screen',new_screen)
		
		print(f'{1/(time.time()-last)} fps')
		
		last=time.time()

		
		new_screen=cv2.resize(new_screen,(50,50))
		prediction=model.predict([new_screen.reshape(-1,50,50,1)])
		
		if(int(prediction[0][0])==0):
			kb.press(Key.space)
			kb.release(Key.space)
			time.sleep(0.1)
		
		current=time.time()
			
		if cv2.waitKey(1) & 0xFF==ord('q'):
			cv2.destroyAllWindows()
			break
		
main()
		
