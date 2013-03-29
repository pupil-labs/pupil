import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
import sys
import scipy.cluster.vq as vq


########################################################################################################
# SET PARAMETERS:
########################################################################################################
"""
ransac is performed on points_infile
if successful the inliers are written to points_outfile
and a backup copy of points_infile is saved as points_backup_file (in case points_infile==points_outfile)
the computed coefficients are written to coeff_outfile
"""
#points_infile='cal_pt_cloud_529.npy'
#points_infile='cal_pt_cloud_good.npy'
points_infile='cal_pt_cloud.npy'
points_outfile='cal_pt_cloud.npy'
points_backup_file='cal_pt_cloud~.npy'
coeff_outfile='calibration_ransac.npy'
nine_point_calibration=True #if points_infile was generated via 9-point calibration method model candidate points are chosen one from each point cluster


"""
ransac parameters:
n - the minimum number of data required to fit the model (7)
k - the number of iterations performed by the algorithm
t - a threshold value for determining when a datum fits a model
d - the number of close data values required to assert that a model fits well to data
	in a 9-point calibration this should be higher than about 88% of total points to ensure that we compute a fit using an even distribution of all 9 point clusters
"""

n=7 #there is also an experimental 9 coefficient model
k=1000
t=6 #inlier threshold in pixels
d=0.9 #minimum number of points required to be inliers (as a ratio of total points)

########################################################################################################




#load data
print "reading data points from %s" %(points_infile)
cal_pt_cloud = np.load(points_infile)
n_points=len(cal_pt_cloud)


# the matrix M is precomputed with all of the eye points, their powers and the corresponding world points

#7 coefficient model
if n==7:
	X=cal_pt_cloud[:,0]
	Y=cal_pt_cloud[:,1]
	XX=X*X
	YY=Y*Y
	XY=X*Y
	XXYY=XX*YY
	Ones=np.ones(n_points)	
	ZX=cal_pt_cloud[:,2]
	ZY=cal_pt_cloud[:,3]
	M=np.array([X,Y,XX,YY,XY,XXYY,Ones,ZX,ZY]).transpose()

#9 coefficient model
if n==9:
	X=cal_pt_cloud[:,0]
	Y=cal_pt_cloud[:,1]
	XX=X*X
	YY=Y*Y
	XY=X*Y
	XXYY=XX*YY
	XXY=XX*Y
	YYX=YY*X
	Ones=np.ones(n_points)	
	ZX=cal_pt_cloud[:,2]
	ZY=cal_pt_cloud[:,3]
	M=np.array([X,Y,XX,YY,XY,XXYY,XXY,YYX,Ones,ZX,ZY]).transpose()

if nine_point_calibration:
	#use kmeans to partition points into 9 clusters
	#point minimum and maximums are used to create and even spaced 3x3 grid of points to use as the initial centroids for kmeans
	mg=np.mgrid[min(M[:,0]) : max(M[:,0]):3j, min(M[:,1]):max(M[:,1]):3j]
	initial_centroids = np.hstack((np.c_[np.ravel(mg[0])],np.c_[np.ravel(mg[1])]))
	centroids,_=vq.kmeans(M[:,0:2], initial_centroids)
	code,_ = vq.vq(M[:,0:2], centroids)

	#create 9 arrays of clustered points for random point generation
	clusters=np.empty(9,dtype='object')
	for j in range(0,9):
		clusters[j]=M[code==j]


#init
i=0
best_error=sys.float_info.max
candidates_found=0

#iterate k times
while i<k:
	
	# select n points to define a candidate model
	if nine_point_calibration:
		#generate m by randomly selecting one point from each cluster
		m=clusters[0][np.random.randint(len(clusters[0]))]
		for j in range(1,9):
			m=np.vstack((m,clusters[j][np.random.randint(len(clusters[j]))]))
		np.random.shuffle(m) # shuffle so that if n<9 model we don't only sample from the first n clusters
	else:
		m=np.vstack((random.sample(M,n)))


	# compute a candidate model based on m[:n,:n]
	try:
		# raises np.linalg.LinAlgError if m is singular
		# this can happen if our random choice of data points is poorly distributed, eg. from the same cluster in a 9-point calibration
		# if m[:n,:n] is singular the we iterate again without counting this pass as one of our k iterations
		cx=np.linalg.solve(m[:n,:n], m[:n,n])	
		cy=np.linalg.solve(m[:n,:n], m[:n,n+1])
	except np.linalg.LinAlgError:
		continue
	else:
		i+=1

	#compute model error for each data point
	err_x=(np.dot(M[:,:n],cx)-M[:,n])*320
	err_y=(np.dot(M[:,:n],cy)-M[:,n+1])*240
	err_dist=np.sqrt(err_x*err_x + err_y*err_y)
	
	#select the inliers
	inliers=M[err_dist<t]
		
	# if this model doesn't have enough inliers then move on
	if(len(inliers)<n_points*d):
		continue

	#store the outliers too, just for display later
	outliers=M[err_dist>=t]

	#compute a model based on all of the inliers using SVD
	U,w,Vt = np.linalg.svd(inliers[:,:n],full_matrices=0)
	V = Vt.transpose()
	Ut = U.transpose()
	pseudINV = np.dot(V, np.dot(np.diag(1/w), Ut))
	cx = np.dot(pseudINV, inliers[:,n])
	cy = np.dot(pseudINV, inliers[:,n+1])

	#compute model error
	err_x=(np.dot(inliers[:,:n],cx)-inliers[:,n])*320
	err_y=(np.dot(inliers[:,:n],cy)-inliers[:,n+1])*240
	err_dist=np.sqrt(err_x*err_x + err_y*err_y)
	err_mean=np.sum(err_dist)/len(err_dist)
	err_rms=math.sqrt(np.sum(err_dist*err_dist)/len(err_dist))

	#see if it's an improvement
	if err_rms < best_error:
		best_error=err_rms

		candidates_found+=1
		best_rms_error=err_rms
		best_mean_error=err_mean
		best_inliers=inliers
		best_outliers=outliers
		best_cx=cx
		best_cy=cy		


# all done iterating
if best_error < sys.float_info.max:
	# success!!!
	# display and store results
	print "candidate models found:", candidates_found
	print "rms error:", best_rms_error
	print "mean error:", best_mean_error
	print "model computed using", len(best_inliers), "inliers of", len(M), "points"

	#for reference, compute improvement over straight SVD using all data points
	U,w,Vt = np.linalg.svd(M[:,:n],full_matrices=0)
	V = Vt.transpose()
	Ut = U.transpose()
	pseudINV = np.dot(V, np.dot(np.diag(1/w), Ut))
	cx = np.dot(pseudINV, M[:,n])
	cy = np.dot(pseudINV, M[:,n+1])
	#compute model error
	err_x=(np.dot(M[:,:n],cx)-M[:,n])*320
	err_y=(np.dot(M[:,:n],cy)-M[:,n+1])*240
	err_dist=np.sqrt(err_x*err_x + err_y*err_y)
	err_mean=np.sum(err_dist)/len(err_dist)
	err_rms=math.sqrt(np.sum(err_dist*err_dist)/len(err_dist))
	print "RANSAC elimination of %d outliers reduced rms error by %.2f percent" %(len(best_outliers), 100*(1-(best_rms_error/err_rms)))

	#save backup if necessary
	if points_infile==points_outfile:		
		print "saving a copy of %s to %s" %(points_infile, points_backup_file)
		np.save(points_backup_file, cal_pt_cloud)

	#save inliers
	print "saving inlier points to %s" %(points_outfile)
	inlier_points=np.hstack((best_inliers[:,0:2],best_inliers[:,n:n+2]))
	np.save(points_outfile, inlier_points)

	#save coefficients
	print "saving coefficients to %s" %(coeff_outfile)
	np.save(coeff_outfile, np.hstack((best_cx,best_cy)))



	#plot eye points
	eye_points_plot=plt.figure()
	plt.title("%s eye points (%d inliers of %d points total), %d coefficient model\n%d RANSAC iterations found %d candidate models\n" %(points_infile, len(best_inliers), len(M), n*2, i, candidates_found), size=10)
	if nine_point_calibration:
		#draw a circle at each centroid whose size is proportional to the number of inlier points in that cluster
		code,_ = vq.vq(best_inliers[:,0:2], centroids)
		for j in range(0,9):
			size=100*len(best_inliers[code==j])
			plt.scatter(centroids[j,0],centroids[j,1], c='w', s=size)
	plt.scatter(best_outliers[:,0],best_outliers[:,1], c='y')
	plt.scatter(best_inliers[:,0],best_inliers[:,1], c='b')

	#plot residuals
	residual_plot=plt.figure()
	plt.title("%s residuals, %d coefficient model, %.2f pixel inlier threshold\nrms error=%f, mean error=%f (in pixels)" %(points_infile, n*2, t, best_rms_error, best_mean_error), size=10)
	err_x=(np.dot(best_inliers[:,:n],best_cx)-best_inliers[:,n])*320
	err_y=(np.dot(best_inliers[:,:n],best_cy)-best_inliers[:,n+1])*240
	plt.scatter(err_x,err_y, c='b')
	err_x=(np.dot(best_outliers[:,:n],best_cx)-best_outliers[:,n])*320
	err_y=(np.dot(best_outliers[:,:n],best_cy)-best_outliers[:,n+1])*240
	plt.scatter(err_x,err_y, c='y')

	plt.show()

else:
	print "failed to find a candidate model after %d iterations." %(i)

		