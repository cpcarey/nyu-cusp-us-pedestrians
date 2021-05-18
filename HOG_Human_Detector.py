# Modified by Chris Carey (cpc9982). This program detects persons in the frame
# and matches them to the closest persons detected in the previous frame, within
# given spatiotemporal thresholds. It displays the velocity and speed of
# matched persons between frames. Velocity is used to project the expected
# position of a person in the next frame in order to match. Velocities of the
# past N frames are collected to produce a mean velocity in order to reduce
# the effect of jitter.

# import the necessary packages
import numpy as np
import cv2
import math
import matplotlib.path as mpl_path
import constants

from person import Person
from roi_match import RoiMatch
from inter_column_match import InterColumnMatch

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture('./videos/demo.mp4')
#cap = cv2.VideoCapture('./videos/PXL_20210512_154130481.mp4')
cap.set(cv2.CAP_PROP_FPS, 25)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (540,960))


persons = []
persons_map = {}
next_id = 0
no_match_map = {}
frame_index = 0
bad = 0

CROP_LEFT = 420
CROP_RIGHT = 860
CROP_TOP = 0
CROP_BOTTOM = 350
PERSON_HEIGHT_MAX = 250

roi = [
    [80, 0],
    [347, 0],
    [347, 208],
    [80, 275],
]
roi_bounds = mpl_path.Path(np.array(roi))
roi_matches_in = []
roi_matches_out = []


def is_person_id_in_roi(id):
    for match in roi_matches_in:
        if id == match.person.id:
            return True
    return False

def get_roi_match(id):
    for match in roi_matches_in:
        if id == match.person.id:
            return match
    return None

def extend_roi_match(person):
    match = get_roi_match(person.id)
    match.path.append(person.centroid)

inter_column_matches = []

def mark_roi_matches():
    for roi_match in roi_matches_in:
        if len(roi_match.path) < constants.ROI_MATCH_PATH_THRESHOLD:
            continue

        if not roi_match.marked:
            N = 5
            path_start = roi_match.path[0:N]
            start_centroid = (int(round(np.sum([p[0] for p in path_start]) / N)),
                              int(round(np.sum([p[1] for p in path_start]) / N)))

            start_time = roi_match.times[0]
            end_centroid = roi_match.path[1]
            end_time = roi_match.times[1]
            entered = False
            splice_index = -1

            for index, time in enumerate(roi_match.times):
                if (time - start_time >= constants.ROI_MATCH_TIME_THRESHOLD and
                    index >= constants.ROI_MATCH_PATH_THRESHOLD):
                    entered = True
                    end_time = time

                    path_end = roi_match.path[index + 1 - N: index + 1]
                    end_centroid = (int(round(np.sum([p[0] for p in path_end]) / N)),
                                    int(round(np.sum([p[1] for p in path_end]) / N)))

                    splice_index = index
                    break

            if entered:
                roi_match.marked = True
                inter_column_match = InterColumnMatch(roi_match,
                                                      start_centroid,
                                                      end_centroid,
                                                      start_time,
                                                      end_time)
                inter_column_matches.append(inter_column_match)
                print(inter_column_match)

run = True
while(True):
    # Capture frame-by-frame
    if run:
        ret, frame = cap.read()

    # resizing for faster detection
    try:
        frame = cv2.resize(frame, (960, 540))
        #if frame_index == 400:
        #    raise Exception
    except:
        print('Analysis Complete: ')
        run = False
        break

    crop = frame[CROP_TOP:CROP_BOTTOM, CROP_LEFT:CROP_RIGHT]
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(crop, winStride=(8,8) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    persons = []
    matched_persons = []

    unmatched_persons = []
    unmatched_persons_map = {}

    # Create persons for each box, to be matched with persons detected
    # previously.
    valid_boxes = []
    for index, box in enumerate(boxes):
        (xA, yA, xB, yB) = box
        # display the detected boxes in the colour picture
        cv2.rectangle(crop, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)

        if yB - yA < PERSON_HEIGHT_MAX:
            valid_boxes.append(box)

    boxes = valid_boxes
    valid_boxes = []
    for box1 in boxes:
        for box2 in boxes:
            valid = True
            # Reject boxes that fully encompass another box.
            if (box1[0] < box2[0] and
                box1[1] < box2[1] and
                box1[2] > box2[2] and
                box1[3] > box2[3]):
                valid = False
                break
        if valid:
            valid_boxes.append(box1)

    for index, box in enumerate(valid_boxes):
        person = Person(box, index)
        unmatched_persons_map[index] = person
        unmatched_persons.append(person)


    # Store the displacements from each previously matched person to each
    # person detected in the current frame.
    displacement_matrix = [[person.get_displacement(prev_person)
                            for prev_person in persons_map.values()]
                           for person in unmatched_persons]

    # Sort each row, then sort rows. This will surface the smallest
    # displacements at the top and left of the matrix, i.e. the closest matches.
    if (len(displacement_matrix) > 0):
        if (len(displacement_matrix[0]) > 0):
            for displacements in displacement_matrix:
                displacements.sort(key=lambda d: d.magnitude)
            displacement_matrix.sort(key=lambda ds: ds[0].magnitude)

    # Iterate through each row to match each detected person in this frame to
    # the closest person in the previous frame, starting with the persons with
    # the smallest displacements.

    while len(displacement_matrix) > 0:
        displacements_for_person = displacement_matrix[0]
        closest_id = -1

        # Remove row used to search.
        displacement_matrix = displacement_matrix[1:]

        # There are previous persons remaining to match. Extract the closest.
        if len(displacements_for_person) > 0:
            displacement = displacements_for_person[0]
            unmatched_person_id = displacement.id0
            closest_prev_person_id = displacement.id1

            matched_person = unmatched_persons_map[unmatched_person_id].clone()

            # Only match the person if the displacement meets the minimum
            # threshold.
            if displacement.magnitude <= constants.MATCH_DISTANCE_THRESHOLD:
                # Save velocities of previous person.
                prev_person = persons_map[closest_prev_person_id]
                prev_velocities = prev_person.prev_velocities

                persons_map[closest_prev_person_id] = matched_person
                matched_person.id = closest_prev_person_id
                matched_person.matched_displacement = displacement

                # Preserve and update velocity buffer.
                matched_person.prev_velocities = prev_velocities
                matched_person.append_velocity()

                matched_persons.append(matched_person)
                del unmatched_persons_map[unmatched_person_id]

                if is_person_id_in_roi(matched_person.id):
                    roi_match = get_roi_match(matched_person.id)
                    if roi_bounds.contains_point(matched_person.centroid):
                        roi_match.extend(matched_person, frame_index)
                    elif not roi_match.locked:
                        roi_match.lock()

            # Remove previous person used from future use in this iteration.
            for index, displacements in enumerate(displacement_matrix):
                displacement_matrix[index] = (
                        [d for d in displacements
                        if d.id0 != closest_prev_person_id and
                        d.id1 != closest_prev_person_id])

    all_persons = matched_persons
    ids = [p.id for p in all_persons]
    mark_roi_matches()

    # Maintain a dictionary of how many frames an unmatched person has gone
    # unmatched.
    for person in unmatched_persons_map.values():
        person.id = next_id
        persons_map[person.id] = person
        all_persons.append(person)
        if roi_bounds.contains_point(person.centroid):
            roi_match = RoiMatch(person, frame_index)
            roi_matches_in.append(roi_match)
        next_id += 1

    for match in inter_column_matches:
        cv2.line(crop,
                 match.start_position,
                 match.end_position,
                 color=(220, 220, 220),
                 thickness=1)

    points = np.array(roi, np.int32).reshape((-1, 1, 2))
    cv2.polylines(crop,
                  [points],
                  isClosed=True,
                  color=(255, 0, 255),
                  thickness=2)

    for person in all_persons:
        person.draw(crop)

    # Remove unmatched persons from re-matching if too much time elapses.
    matched_persons_ids = set([p.id for p in matched_persons])
    for person in list(persons_map.values()):
        if person.id in matched_persons_ids:
            if person.id in no_match_map:
                del no_match_map[person.id]
        else:
            if not person.id in no_match_map:
                no_match_map[person.id] = 0
            no_match_map[person.id] += 1

            if no_match_map[person.id] > constants.NO_MATCH_FRAME_THRESHOLD:
                del persons_map[person.id]
                del no_match_map[person.id]

    # Write the output video
    #out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame',crop)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
