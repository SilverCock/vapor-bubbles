def match_objects(active_objects, detections, frame_idx, tracks, objects_summary, next_id):
    updated_objects = []

    for obj in active_objects:
        found = False
        candidates = [d for d in detections if d['class'] == obj['class'] and not d['matched']]

        if candidates:
            tree = cKDTree([(d['x'], d['y']) for d in candidates])
            dist, idx = tree.query([obj['x'], obj['y']], distance_upper_bound=D_MAX)

            if dist < D_MAX and idx < len(candidates):
                d = candidates[idx]
                d['matched'] = True
                d['id'] = obj['id']
                d['start'] = obj['start']
                updated_objects.append(d)

                if obj['class'] == 'bubble':
                    tracks.append([d['id'], frame_idx, d['x'], d['y'], d['radius']])

                found = True

        if not found:
            objects_summary.append([
                obj['id'], obj['class'], obj['start'],
                frame_idx - 1, frame_idx - obj['start']
            ])
