# -*- coding: utf-8 -*-
"""
Created on Sat May 20 16:10:30 2017

@author: diz
"""
from moviepy.editor import VideoFileClip

thr = 18
tau = 0.95
scales = [3, 4, 5]
heat = np.zeros_like(img[:,:,0]).astype(np.float)
  
video_output = 'out/p5C08sc%d_%.2f_%.1f.mp4' % (345, tau, thr)
clip = VideoFileClip('project_video.mp4')#.subclip(30,45)
#clip = VideoFileClip('E:\\Data\\USA\\Video\\cuts\\multiple_01.avi') 
first_clip = clip.fl_image(process_image)
get_ipython().magic('time first_clip.write_videofile(video_output, audio=False)')
        
#%%