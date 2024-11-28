#Original Code


def _get_approximate_success(prev_rgb, frame, action):
	wheres = np.where(prev_rgb != frame)
	wheres_ar = np.zeros(prev_rgb.shape)
	wheres_ar[wheres] = 1
	wheres_ar = np.sum(wheres_ar, axis=2).astype(bool)
	connected_regions = skimage.morphology.label(wheres_ar, connectivity=2)
	unique_labels = [i for i in range(1, np.max(connected_regions)+1)]
	max_area = -1
	for lab in unique_labels:
		wheres_lab = np.where(connected_regions == lab)
		max_area = max(len(wheres_lab[0]), max_area)
	if (action in ['OpenObject', 'CloseObject']) and max_area > 500:
		success = True
	elif max_area > 100:
		success = True
	else:
		success = False
	return success

def success_for_look(self, action):
    wheres = np.where(self.prev_rgb != self.event.frame)
    wheres_ar = np.zeros(self.prev_rgb.shape)
    wheres_ar[wheres] = 1
    wheres_ar = np.sum(wheres_ar, axis=2).astype(bool)
    connected_regions = skimage.morphology.label(wheres_ar, connectivity=2)
    unique_labels = [i for i in range(1, np.max(connected_regions)+1)]
    max_area = -1
    for lab in unique_labels:
        wheres_lab = np.where(connected_regions == lab)
        max_area = max(len(wheres_lab[0]), max_area)
    if action in ['OpenObject', 'CloseObject'] and max_area > 500:
        success = True
    elif max_area > 100:
        success = True
    else:
        success = False
    return success


def watercourse(preg_rgb, frame, sink_mask, faucet_mask):
	#First get all the diff regions

	#Then among these get one that 