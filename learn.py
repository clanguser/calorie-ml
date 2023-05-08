import numpy as np
import cv2
from create_feature import *
from calorie_calc import *
import csv
from sklearn import svm


svm_params = dict(kernel_type=cv2.ml.SVM_LINEAR, svm_type=cv2.ml.SVM_C_SVC, C=2.67, gamma=5.383)
def training():
	feature_mat = []
	response = []
	for j in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
		for i in range(1,21):
			bruh = r"E:/VIT/Calorie/images/All_Images/"+str(j)+r"_"+str(i)+r".jpg"
			print(bruh)
			fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg(bruh)
			feature_mat.append(fea)
			response.append(float(j))

	trainData = np.float32(feature_mat).reshape(-1,94)
	responses = np.float32(response)

	svm = cv2.ml.SVM_create()
	# svm = svm.SVC()
	svm.train(trainData,cv2.ml.ROW_SAMPLE,responses.astype(int))
	svm.save('svm_data.dat')	

def testing():
	svm_model = cv2.ml.SVM_create()
	svm_model = cv2.ml.SVM_load('svm_data.dat')
	
	feature_mat = []
	response = []
	image_names = []
	pix_cm = []
	fruit_contours = []
	fruit_areas = []
	fruit_volumes = []
	fruit_mass = []
	fruit_calories = []
	skin_areas = []
	fruit_calories_100grams = []
	for j in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
		for i in range(21,26):	
			img_path = "E:/VIT/Calorie/images/Test_Images/"+str(j)+"_"+str(i)+".jpg"
			print(img_path)
			fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg(img_path)
			pix_cm.append(pix_to_cm)
			fruit_contours.append(fcont)
			fruit_areas.append(farea)
			feature_mat.append(fea)
			skin_areas.append(skinarea)
			response.append([float(j)])
			image_names.append(img_path)

	testData = np.float32(feature_mat).reshape(-1,94)
	responses = np.float32(response)
	result = svm_model.predict(testData)
	# result, _ = svm_model.predict(testData, cv2.ml.ROW_SAMPLE)

	mask = result[1]==responses
	#calculate calories
	for i in range(0, len(result[1])):
		volume = getVolume(int(result[1][i]), fruit_areas[i], skin_areas[i], pix_cm[i], fruit_contours[i])
		mass, cal, cal_100 = getCalorie(result[1][i], volume)
		fruit_volumes.append(volume)
		fruit_calories.append(cal)
		fruit_calories_100grams.append(cal_100)
		fruit_mass.append(mass)
		
	#write into csv file
	with open('E:/VIT/Calorie/out.csv', 'w') as outfile:
		writer = csv.writer(outfile)
		data = ["Image name", "Desired response", "Output label", "Volume (cm^3)", "Mass (grams)", "Calories for food item", "Calories per 100 grams"]
		writer.writerow(data)
		for i in range(0, len(result[1])):
			if (fruit_volumes[i] == None):
				data = [str(image_names[i]), str(int(responses[i][0])), str(int(result[1][i][0])), "--", "--", "--", str(fruit_calories_100grams[i])]
			else:
				data = [str(image_names[i]), str(int(responses[i][0])), str(int(result[1][i][0])), str(fruit_volumes[i]), str(fruit_mass[i]), str(fruit_calories[i]), str(fruit_calories_100grams[i])]
			writer.writerow(data)
		outfile.close()
	
	# print op
	# print(fruit_volumes)
	# print("Image name", "Desired response", "Output label", "Volume (cm^3)", "Mass (grams)", "Calories for food item", "Calories per 100 grams")
	# for i in range(0, len(result[1])):
	# 	if (fruit_volumes[i] == None):
	# 		print(str(image_names[i]), str(int(responses[i][0])), str(int(result[1][i][0])), "--", "--", "--", str(fruit_calories_100grams[i]))
	# 	else:
	# 		print(str(image_names[i]), str(int(responses[i][0])), str(int(result[1][i][0])), str(fruit_volumes[i]), str(fruit_mass[i]), str(fruit_calories[i]), str(fruit_calories_100grams[i]))
	
	for i in range(0, len(mask)):
		if mask[i][0] == False:	
			print ("(Actual Reponse)", int(responses[i][0]), "(Output)", int(result[1][i][0]), image_names[i])

	correct = np.count_nonzero(mask)
	print(correct*100.0/result[1].size)

	

if __name__ == '__main__':
	# training()
	# run training() only once to train a model
	testing()


