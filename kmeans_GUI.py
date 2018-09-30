from tkinter import messagebox
from tkinter import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

# saving dataset
def save_dataset():
	clustered_data.to_csv('Clustered_dataset/clustered_dataset.csv')
	messagebox.showinfo('Success','Clustered dataset saved successfully.')

# loading dataset
def upload_dataset():
	global data

	# obtaining dataset name from the GUI
	dataset_name=datasetname_Entry.get()
	dataset_location='Original_dataset/'
	if dataset_name != 'xclara.csv':
		messagebox.showinfo('Fail','No dataset exists with this name.')
	
	# loading dataset as a pandas dataframe
	data=pd.read_csv(dataset_location + dataset_name)
	messagebox.showinfo('Success','Dataset uploaded successfully.')

# applying clustering algorithm
def kmeans_clustering():
	global clustered_data
	
	# creating numpy matrix of the features
	f1 = data[data.columns[0]].values
	f2 = data[data.columns[1]].values

	# concatinating feature matrix
	X = np.array(list(zip(f1, f2)))

	# obtain k input from GUI
	k=int(K_Entry.get())
	
	# X coordinates of random centroids
	C_x = np.random.randint(0, np.max(X)-20, size=k)
	# Y coordinates of random centroids
	C_y = np.random.randint(0, np.max(X)-20, size=k)

	C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
		
	# To store the value of centroids when it updates
	C_old = np.zeros(C.shape)

	# Cluster Lables(0, 1, 2)
	clusters = np.zeros(len(X))

	# Error func. - Distance between new centroids and old centroids
	error = dist(C, C_old, None)

	# Loop will run till the error becomes zero
	while error != 0:
   		# Assigning each value to its closest cluster
		for i in range(len(X)):
			distances = dist(X[i], C)
			cluster = np.argmin(distances)
			clusters[i] = cluster
    		# Storing the old centroid values
		C_old = deepcopy(C)
    		# Finding the new centroids by taking the average value
		for i in range(k):
			points = [X[j] for j in range(len(X)) if clusters[j] == i]
			C[i] = np.mean(points, axis=0)
		error = dist(C, C_old, None)
	
	# creating new feature in the dataset containing cluster labels
	clustered_data = data.assign(cluster=pd.Series(clusters).values)
	messagebox.showinfo('Success','Dataset clustered successfully.')

	# Data Visualization
	colors = ['r', 'g', 'b', 'y', 'c', 'm']
	fig, ax = plt.subplots()
	for i in range(k):
        	points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        	ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
	ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
	plt.show()

# calculating euclidian distance
def dist(a, b, ax=1):
	return np.linalg.norm(a - b, axis=ax)

# application exit status
def close_app():
	msg=messagebox.askquestion("Exit Application","Are you sure you want to quit ?")
	if msg=='yes':
		root.destroy()
	else:
		messagebox.showinfo('Return','You will now return to the application screen')
        
root=Tk()

# setting background image for the app
background_image=PhotoImage(file='Image/image.png')
background_label = Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# app title
root.title("CLUSTERING WIZARD")

# taking entries from the user
K_Entry = Entry(root, width=30,bg='pale goldenrod',borderwidth=1, relief="solid")
datasetname_Entry=Entry(root,width=30,bg='pale goldenrod',borderwidth=1, relief="solid")

# initializing button events
upload_button=Button(text="UPLOAD DATASET",width=20,height=1,bg="grey",fg="white",borderwidth=3,relief="solid",command=upload_dataset)
compute_button=Button(text="CLUSTER DATASET",width=20,height=1,bg="grey",fg="white",borderwidth=3,relief="solid",command=kmeans_clustering)
save_button=Button(text="SAVE CLUSTERED DATASET",width=20,height=1,bg="grey",fg="white",borderwidth=3,relief="solid",command=save_dataset)
exit_button=Button(text="EXIT",width=20,height=1,bg="grey",fg="white",borderwidth=3,relief="solid",command=close_app)

# initializing labels
k_label=Label(root,text="Please enter the value of K ",font=('Times new Roman',13), bg='tan1',borderwidth=2, relief="solid")
datasetname_label=Label(root,text="Please enter the name of the dataset ",font=('Times new Roman',13),bg='tan1',borderwidth=2, relief="solid")

# packing widgets on the application grid
datasetname_label.grid(row=1,column=1)
datasetname_Entry.grid(row=1,column=3)
k_label.grid(row=3,column=1,sticky=E)
K_Entry.grid(row=3,column=3)
upload_button.grid(row=5,column=3)
compute_button.grid(row=7,column=3)
save_button.grid(row=9,column=3)
exit_button.grid(row=11,column=3)

# setting application resolution
root.geometry("560x280")

# account for empty rows in the grid
root.grid_rowconfigure(0,minsize=15)
root.grid_rowconfigure(2,minsize=15)
root.grid_rowconfigure(4,minsize=15)
root.grid_rowconfigure(6,minsize=15)
root.grid_rowconfigure(8,minsize=15)
root.grid_rowconfigure(10,minsize=15)

# account for empty columns in the grid
root.grid_columnconfigure(0,minsize=15)
root.grid_columnconfigure(2,minsize=15)

# loop to render the application
root.mainloop()
