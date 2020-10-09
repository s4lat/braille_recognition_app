from br_utils import prepFilter1, prepFilter2, non_max_suppression_fast
from tensorflow import keras
import cv2, argparse
import numpy as np
import sys

DEBUG = True

isFinded = False
isEditing = False
isValid = False
isTranslated = False

MODEL = keras.models.load_model('./model')
def noth():
	pass
if DEBUG:
	cv2.namedWindow('filter')

cv2.namedWindow('warp')
cv2.namedWindow('out')

parser = argparse.ArgumentParser()
parser.add_argument('--img', dest='img')

args = parser.parse_args()

if not args.img:
	url = "http://192.168.0.131:8080/video"
	cap = cv2.VideoCapture(url)
else:
	frame = cv2.imread(args.img)



def draw_doc_box(img, pts):
	frame = img.copy()

	frame = cv2.polylines(frame, pts[np.newaxis], True, (0, 255, 0), 2)
	cv2.circle(frame, tuple(pts[0]), 10, (0, 255, 0), -1)
	cv2.circle(frame, tuple(pts[1]), 10, (0, 0, 255), -1)
	cv2.circle(frame, tuple(pts[2]), 5, (0, 255, 0), -1)
	cv2.circle(frame, tuple(pts[3]), 5, (0, 0, 255), -1)

	return frame

def find_biggest_rect(contours):
	biggest_c = None
	for c in contours:
		area = cv2.contourArea(c)
		epsilon = 0.1*cv2.arcLength(c, False)
		approx = cv2.approxPolyDP(c,epsilon, True)
		if len(approx) == 4:
			biggest_c = approx

	if biggest_c is not None:
		return biggest_c

def sort_clockwise(c):
	diffs = np.zeros(3)
	for i, p in enumerate(c[1:]):
		diffs[i] = (c[0, 1] - p[1])**2

	np_idx = np.argmin(diffs)
	if np_idx == 0:
		lt = c[1]
		rt = c[0]
		rd = c[3]
		ld = c[2]
	else:
		lt = c[0]
		rt = c[3]
		rd = c[2]
		ld = c[1]

	return np.array([lt, rt, rd, ld])


p_to_edit_idx = 0
def edit_selection(event, x, y, flags, param):
	global isFinded, isValid, isEditing, pts, p_to_edit_idx

	if not (isFinded and not isValid):
		return

	if event == cv2.EVENT_LBUTTONDOWN:
		isEditing = True
		nearest_p_idx = np.argmin(np.sum((np.array([x, y]) - pts) ** 2, 1))
		p_to_edit_idx = nearest_p_idx

	elif event == cv2.EVENT_MOUSEMOVE:
		if isEditing:
			pts[p_to_edit_idx] = [x, y]

	elif event == cv2.EVENT_LBUTTONUP:
		isEditing = False

cv2.setMouseCallback("out", edit_selection)

while True:
	k = cv2.waitKey(1) & 0xFF
	if k == ord('q'):
		break
	elif k == ord('f'):
		isFinded = True

	elif k == ord('h'):
		isFinded = True
		if frame is not None:
			pts = np.array([[frame.shape[1]//5, frame.shape[0]//5],
							[(frame.shape[1]//5)*4, frame.shape[0]//5],
							[(frame.shape[1]//5)*4, (frame.shape[0]//5)*4],
							[frame.shape[1]//5, (frame.shape[0]//5)*4]])
			frame_w_box = draw_doc_box(frame, pts)
			cv2.imshow('out', frame_w_box)

	elif k == ord('r'):
		isFinded = False
		isValid = False

	elif k == ord('p') and isFinded:
		isValid = True
		scale_fac = np.float32([src.shape[1] / frame.shape[1],
		 					  src.shape[0] / frame.shape[0]])

		pts0 = np.float32(pts) * scale_fac

		w,h = (np.sqrt(np.sum((pts0[1] - pts0[0])**2)),
			   np.sqrt(np.sum((pts0[1] - pts0[2])**2)))
		w,h = int(w), int(h)

		pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

		mat = cv2.getPerspectiveTransform(pts0, pts1)
		warp = cv2.warpPerspective(src, mat, (w, h))
		warp = cv2.resize(warp, (warp.shape[1]*2, warp.shape[0]*2))
		cv2.imwrite('./warp.jpg', warp)
		cv2.imshow('warp', warp)

	elif k == ord('b') and isValid:
		gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
		filtered = prepFilter2(gray)
		filtered = prepFilter1(filtered)
		cv2.imwrite('./filtered.jpg', filtered)

		segment_coords = []
		segments = []
		coords = []
		for y in range(4, len(filtered)):
			for x in range(4, len(filtered[y])):
				if filtered[y, x]>5:
					cv2.circle(filtered, (x+3, y+3), (7), color=(0, 255, 0), thickness=-1)
					cut = gray[y-4:y+12, x-4:x+12]
					segment = np.zeros((16, 16))
					segment[:cut.shape[0], :cut.shape[1]] = gray[y-4:y+12, x-4:x+12]

					segments.append(segment)
					segment_coords.append([x, y])

		segments = np.array(segments).reshape((len(segments), 16, 16, 1))
		segment_coords = np.array(segment_coords)

		preds = MODEL.predict(segments)
		preds = preds.reshape(len(preds))
		coords = segment_coords[preds > .7]

		boundingBoxes = np.empty((coords.shape[0], 4))
		boundingBoxes[:,0], boundingBoxes[:,1], boundingBoxes[:,2], boundingBoxes[:,3] = (
			np.array([coords[:,0]-4, coords[:,1]-4, coords[:,0]+12, coords[:,1]+12])
		)

		pick = non_max_suppression_fast(boundingBoxes, 0.1)
		print(len(pick))
		for (startX, startY, endX, endY) in pick:
			cv2.rectangle(warp, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0))

		cv2.imshow('warp', warp)

	if isEditing:
		frame_w_box = draw_doc_box(frame, pts)
		cv2.imshow('out', frame_w_box)

	if not isFinded:
		if not args.img:
			ret, frame = cap.read()

			if not ret:
				continue

		src = frame.copy()
		if not args.img:
			frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
		else:
			isFinded = True

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (5, 5), 1)
		binary = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	            cv2.THRESH_BINARY, 51, -3)

		contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		if len(contours) != 0:
			contours = sorted(contours, key=cv2.contourArea)[-5::1]
			biggest_c = find_biggest_rect(contours)

			if biggest_c is not None:
				biggest_c = biggest_c.reshape((4, 2))
				pts = sort_clockwise(biggest_c)
				frame_w_box = draw_doc_box(frame, pts)
				cv2.imshow('out', frame_w_box)
			else:
				cv2.imshow('out', frame)

		else:
			cv2.imshow('out', frame)


		if DEBUG:
			cv2.imshow('filter', binary)

	
if not args.img:
	cap.release()
cv2.destroyAllWindows()