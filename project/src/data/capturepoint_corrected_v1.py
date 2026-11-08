"""Explicit OpenCVL mast3rpnp labels for candidate holdout preparation.

All outputs are evaluator metadata. No fallback, policy, image or model code.
"""
import math
from .capturepoint_intake_v1 import safe_relative

EARTH_RADIUS_M = 6371008.8


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Finite numeric field required; booleans rejected')
    return float(value)


def latlon(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('Explicit geographic center required')
    lat, lon = map(number, value)
    if not -90 < lat < 90 or not -180 <= lon <= 180:
        raise ValueError('Invalid geographic center')
    return [lat, lon]


def corrected_pose(label):
    sid = label.get('mapil_image_id')
    if not isinstance(sid, str) or not sid.isdigit():
        raise ValueError('Exact stable Mapillary image ID required')
    fields = [label.get(k) for k in ('mapil_image', 'ground_image') if label.get(k)]
    if not fields or len(set(fields)) != 1:
        raise ValueError('Missing or ambiguous ground image')
    ground = safe_relative(fields[0]); aerial = safe_relative(label['aerial_image'])
    if ground.rsplit('/', 1)[-1] != sid+'.png' or aerial.rsplit('/', 1)[-1] != sid+'.png':
        raise ValueError('Pair paths must match the selected stable ID')
    # These three fields must all come from the same explicit namespace.
    dx, dy, heading = [number(label[k]) for k in ('dx_mast3rpnp', 'dy_mast3rpnp', 'heading_mast3rpnp')]
    resolution = number(label['resolution'])
    if resolution <= 0:
        raise ValueError('Positive ground sampling distance required')
    return dict(source_key='mapillary:'+sid, image_id=sid, ground_member=ground, aerial_member=aerial,
                pose_variant='mast3rpnp', dx=dx, dy=dy, heading=heading,
                resolution=resolution, aerial_latlon=latlon(label['aerial_latlon']),
                pose_semantics='published camera point; not visible landmark',
                camera_latlon_not_supplied=True, policy_input=False,
                absolute_survey_verified=False, calibrated_photo_admitted=False)


def geometry(label, width, height):
    if type(width) is not int or type(height) is not int or not 5 <= width <= 4096 or not 5 <= height <= 4096:
        raise ValueError('Bounded native dimensions required')
    p = corrected_pose(label)
    x, y = width/2+p['dx'], height/2-p['dy']
    xe, ye = [[i*n//5 for i in range(6)] for n in (width, height)]
    inside = 0 <= x < width and 0 <= y < height
    row = next((i for i in range(5) if ye[i] <= y < ye[i+1]), None) if inside else None
    col = next((i for i in range(5) if xe[i] <= x < xe[i+1]), None) if inside else None
    # An aerial-center circle does not require inventing a corrected camera GPS.
    radius = math.hypot(width*p['resolution']/2, height*p['resolution']/2)
    return dict(x=x, y=y, inside=inside, row=row, col=col, index=row*5+col if inside else None,
                x_edges=xe, y_edges=ye, footprint_m=[width*p['resolution'], height*p['resolution']],
                reference_center_latlon=p['aerial_latlon'], reference_radius_m=radius,
                pixel_formula='x=W/2+dx_mast3rpnp; y=H/2-dy_mast3rpnp',
                coordinate_scope='dataset north-up/scale; not independently surveyed', policy_input=False)


def distance(a, b):
    la, lo = map(math.radians, latlon(a)); lb, lp = map(math.radians, latlon(b))
    t = math.sin((lb-la)/2)**2+math.cos(la)*math.cos(lb)*math.sin((lp-lo)/2)**2
    return 2*EARTH_RADIUS_M*math.asin(math.sqrt(max(0., min(1., t))))


def separation(candidates, known, buffer_m=500.):
    """Conservative circles for the declared ledger only; no global novelty claim."""
    if number(buffer_m) < 0:
        raise ValueError('Nonnegative buffer required')
    checks = []
    unknown = [r['source_key'] for r in known if r.get('center_latlon') is None]
    for i, candidate in enumerate(candidates):
        for other in [*known, *candidates[:i]]:
            identical = candidate['source_key'] == other['source_key']
            shared = bool(set(candidate.get('image_hashes', [])) & set(other.get('image_hashes', [])))
            if other.get('center_latlon') is None:
                checks.append(dict(a=candidate['source_key'], b=other['source_key'], passed=False,
                                   reason='same identity/image' if identical or shared else 'missing geographic ledger fields'))
                continue
            ra, rb = number(candidate['radius_m']), number(other['radius_m'])
            if min(ra, rb) < 0:
                raise ValueError('Nonnegative footprint radii required')
            d = distance(candidate['center_latlon'], other['center_latlon'])
            gap = d-ra-rb
            checks.append(dict(a=candidate['source_key'], b=other['source_key'], center_distance_m=d,
                               conservative_gap_m=gap, same_identity=identical, shared_image=shared,
                               passed=not identical and not shared and gap >= buffer_m))
    return dict(checks=checks, declared_known_sources=len(known), missing_geographic_sources=unknown,
                buffer_m=buffer_m, declared_ledger_separation_passed=bool(candidates) and all(c['passed'] for c in checks),
                global_geographic_independence_verified=False, pretraining_overlap='unknown',
                formal_confirmation_passed=False, policy_input=False)


def public_packet(packet):
    """Only an ordinary image pair and already-public state can enter a policy."""
    expected = {'target_rgb', 'current_rgb', 'current_index', 'visited_indices', 'remaining_budget'}
    if type(packet) is not dict or set(packet) != expected:
        raise ValueError('Only ordinary RGB and public state; no evaluator metadata')
    current, remaining = packet['current_index'], packet['remaining_budget']
    visited = packet['visited_indices']
    if type(current) is not int or not 0 <= current < 25 or type(remaining) is not int or not 0 <= remaining <= 10:
        raise ValueError('Invalid public state')
    if not isinstance(visited, list) or any(type(v) is not int or not 0 <= v < 25 for v in visited) or current not in visited:
        raise ValueError('Invalid visited history')
    return dict(packet)
