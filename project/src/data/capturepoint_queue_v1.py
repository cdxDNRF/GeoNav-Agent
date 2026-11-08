"""CV042 data-only camera-position queues. No model or environment execution."""
import math

OLD_IDS = {'10000258569988630', '1000015350919599', '1000081124702485'}
ACTIONS = ('up', 'right', 'down', 'left')
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def select_pairs(admission, directory):
    entries = {e['name']: e for e in directory['entries']}
    pairs = []
    for member in sorted(admission['public_metadata']['city_label_members'], key=lambda x: x['member']['name']):
        prefix = member['member']['name'].rsplit('/', 1)[0]
        rows = sorted(member['records'], key=lambda x: x['mapil_image'])
        if len(rows) != 3 or rows[0]['mapil_image_id'] not in OLD_IDS:
            raise ValueError('Expected three frozen rows/city and the original representative first')
        for order, row in enumerate(rows):
            if not row['pose_mapilopensfm_available']:
                raise ValueError('No fallback from the frozen corrected pose')
            pair_id = row['city'] + '_' + row['mapil_image_id']
            pairs.append(dict(pair_id=pair_id, city=row['city'], id=row['mapil_image_id'],
                              role='fit_preparation' if order < 2 else 'development',
                              previously_consumed=row['mapil_image_id'] in OLD_IDS,
                              row_index=member['records'].index(row), label_member=member['member']['name'],
                              label=row, members={kind: entries[prefix+'/'+row[key]]
                                                  for kind, key in [('ground', 'mapil_image'), ('aerial', 'aerial_image')]}))
    if len(pairs) != 9 or sum(p['role'] == 'fit_preparation' for p in pairs) != 6:
        raise ValueError('Frozen 6 fit-preparation / 3 development split required')
    fit = [p for p in pairs if p['role'] == 'fit_preparation']
    for p in pairs:
        p['wrong_donor_id'] = next(q['pair_id'] for q in fit
                                  if q['city'] == p['city'] and q['pair_id'] != p['pair_id'])
    return pairs


def distance_m(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, [*a, *b])
    v = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371008.8 * 2 * math.asin(math.sqrt(min(1., max(0., v))))


def point_and_cell(label, width, height, variant='opensfm', k=5):
    if variant not in ('raw', 'opensfm') or width < k or height < k:
        raise ValueError('Supported pose and positive raster required')
    x = width/2 + float(label['dx_mapil'+variant])
    y = height/2 - float(label['dy_mapil'+variant])
    if not math.isfinite(x+y) or not 0 <= x < width or not 0 <= y < height:
        raise ValueError('Camera point outside image; never clip a label')
    xe = [i*width//k for i in range(k+1)]
    ye = [i*height//k for i in range(k+1)]
    col = next(i for i in range(k) if xe[i] <= x < xe[i+1])
    row = next(i for i in range(k) if ye[i] <= y < ye[i+1])
    res = float(label['resolution'])
    if not math.isfinite(res) or res <= 0:
        raise ValueError('Metric scale required')
    return dict(x=x, y=y, row=row, col=col, index=row*k+col, x_edges=xe, y_edges=ye,
                footprint_m=[width*res, height*res],
                label_radius_m=max(math.hypot(xx-x, yy-y)*res
                                   for xx in (0, width) for yy in (0, height)))


def separation(pairs, geometries=None):
    checks = []
    for i, p in enumerate(pairs):
        for q in pairs[i+1:]:
            d = distance_m(p['label']['ground_latlon_mapilopensfm'], q['label']['ground_latlon_mapilopensfm'])
            # Before decoding, the decoder's 4096px bound supplies a conservative circle.
            def radius(v):
                if geometries is not None:
                    return geometries[v['pair_id']]['label_radius_m']
                r = v['label'];return (4096/math.sqrt(2)+abs(r['dx_mapilopensfm'])+abs(r['dy_mapilopensfm']))*r['resolution']
            gap = d-radius(p)-radius(q)
            checks.append(dict(a=p['pair_id'], b=q['pair_id'], camera_distance_m=d,
                               conservative_footprint_gap_m=gap, cross_role=p['role'] != q['role'],
                               at_least_500m_by_dataset_labels=gap >= 500))
    return dict(checks=checks, all_pass=all(x['at_least_500m_by_dataset_labels'] for x in checks),
                minimum_camera_distance_m=min(x['camera_distance_m'] for x in checks),
                minimum_footprint_gap_m=min(x['conservative_footprint_gap_m'] for x in checks),
                basis='dataset corrected camera coordinates + bounded footprint circles; not independent surveying',
                city_unseen=False, geographic_units=len(pairs), cities=len({p['city'] for p in pairs}))


def make_tasks(pairs, geometries, k=5, budget=10):
    tasks, truth = [], []
    for p in pairs:
        geo = geometries[p['pair_id']];goal = geo['index'];gr, gc = divmod(goal, k)
        for start in range(k*k):
            if start == goal:
                continue
            sr, sc = divmod(start, k);distance = abs(sr-gr)+abs(sc-gc)
            if distance > budget:
                raise ValueError('Frozen task is unreachable even on a shortest path')
            tid = f"{p['pair_id']}_s{start:02d}"
            tasks.append(dict(task_id=tid, pair_id=p['pair_id'], role=p['role'],
                              start_index=start, grid_size=k, budget=budget,
                              wrong_donor_id=p['wrong_donor_id'],
                              execution_scope='evaluator manifest, never an Agent observation'))
            progress = []
            for action, (dr, dc) in zip(ACTIONS, DELTAS):
                rr, cc = sr+dr, sc+dc
                if 0 <= rr < k and 0 <= cc < k and abs(rr-gr)+abs(cc-gc) < distance:
                    progress.append(action)
            truth.append(dict(task_id=tid, goal_index=goal, goal_xy=[geo['x'], geo['y']],
                              initial_distance_grid=distance, progress_actions_at_start=progress,
                              use='supervision/evaluation only; no inference input'))
    return tasks, truth


def public_state(current_index, visited_indices, remaining_budget, k=5):
    """Whitelist the non-image policy state; no task/pair/goal/role/source fields."""
    if any(type(v) is not int for v in (current_index, remaining_budget, k)) or k != 5:
        raise ValueError('Integer state for the frozen 5x5 protocol required')
    if not 0 <= current_index < k*k or not 0 <= remaining_budget <= 10:
        raise ValueError('Invalid public state')
    visited = sorted(set(visited_indices))
    if any(type(i) is not int or not 0 <= i < k*k for i in visited) or current_index not in visited:
        raise ValueError('Visited history must contain exactly valid revealed cells')
    return dict(current_index=current_index, grid_size=k, visited_indices=visited,
                remaining_budget=remaining_budget)
