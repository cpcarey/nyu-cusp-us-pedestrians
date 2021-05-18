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
import constants

from person import Person
from track import Track, TrackClassification
from inter_column_match import InterColumnMatch
from roi import Roi

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

active_tracks = []
expired_tracks = []
roi = Roi()

def get_track(id, tracks):
    for match in tracks:
        if id == match.person.id:
            return match
    return None

def extend_track(person):
    match = get_track(person.id)
    match.path.append(person.centroid)

inter_column_matches = []

def mark_track(track):
    if not track.is_classified():
        if track.classify() == TrackClassification.INSIDE_ROI:
            inter_column_match = InterColumnMatch(track)
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

                track = get_track(matched_person.id, active_tracks)
                if track:
                    track.extend(matched_person, frame_index)

            # Remove previous person used from future use in this iteration.
            for index, displacements in enumerate(displacement_matrix):
                displacement_matrix[index] = (
                        [d for d in displacements
                        if d.id0 != closest_prev_person_id and
                        d.id1 != closest_prev_person_id])

    all_persons = matched_persons
    ids = [p.id for p in all_persons]
    for track in active_tracks:
        mark_track(track)

    # Maintain a dictionary of how many frames an unmatched person has gone
    # unmatched.
    for person in unmatched_persons_map.values():
        person.id = next_id
        persons_map[person.id] = person
        all_persons.append(person)
        track = Track(person, frame_index)
        active_tracks.append(track)
        next_id += 1

    for match in inter_column_matches:
        match.draw(crop)

    roi.draw(crop)

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
                track = get_track(person.id, active_tracks)
                if track:
                    active_tracks.remove(track)
                    expired_tracks.append(track)

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
