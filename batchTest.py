# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:03:43 2017

@author: Anton Varfolomeev
"""

window = 32
thr = 11
tau = 0.96
scales = [3, 4, 5]
from moviepy.editor import VideoFileClip
#from IPython.display import HTML
     
for thr in [ 24, 22, 20]:
    for tau in [0.96, 0.95]:
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        video_output = 'out/p5sc%d_%.2f_%.1f.mp4' % (len(scales), tau, thr)
        clip = VideoFileClip('project_video.mp4') #.subclip(30,45)
        #clip = VideoFileClip('E:\\Data\\USA\\Video\\cuts\\multiple_01.avi') 
        first_clip = clip.fl_image(process_image)
        get_ipython().magic('time first_clip.write_videofile(video_output, audio=False)')
        