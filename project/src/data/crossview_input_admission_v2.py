"""Explicit Mapillary pose metadata adapter; never a policy input or GPS oracle.

V1 ZOD contracts are retained. This adapter does not assume an aerial GPS field,
absolute survey accuracy, actual image dimensions, or a particular camera FoV.
"""
import math
from project.src.data.crossview_input_admission_v1 import (
    K_status, connected_groups, training_readiness, transformed_K,
    validate_public_packet,
)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _latlon(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('latitude/longitude pair required')
    if not all(_finite(v) for v in value) or not -90 <= value[0] <= 90 or not -180 <= value[1] <= 180:
        raise ValueError('invalid geographic coordinates')
    return list(value)


def mapillary_pose(row, *, variant):
    """Select every pose field from ONE explicit namespace, without fallback.

    ground_latlon in the acquired sample is the raw alias. It must not be paired
    silently with corrected dx/dy. The corrected pose is dataset supervision,
    not independently surveyed truth. Heading is retained for evaluator checks
    only and remains forbidden in the ordinary-photo public model packet.
    """
    if variant not in ('raw', 'opensfm'):
        raise ValueError('explicit raw or opensfm pose required')
    suffix = 'mapilraw' if variant == 'raw' else 'mapilopensfm'
    if variant == 'opensfm' and row.get('pose_mapilopensfm_available') is not True:
        raise ValueError('corrected pose unavailable; no raw fallback')
    raw = _latlon(row['ground_latlon_mapilraw'])
    if _latlon(row['ground_latlon']) != raw:
        raise ValueError('legacy ground_latlon is not the documented raw alias')
    position = _latlon(row['ground_latlon_' + suffix])
    dx, dy, heading, resolution = (row['dx_' + suffix], row['dy_' + suffix],
                                    row['heading_' + suffix], row['resolution'])
    if not all(_finite(v) for v in (dx, dy, heading, resolution)) or resolution <= 0:
        raise ValueError('finite pose offsets and positive resolution required')
    ground, aerial = row['mapil_image'], row['aerial_image']
    if not all(isinstance(v, str) and v for v in (ground, aerial)):
        raise ValueError('two exact image member paths required')
    return dict(ground_image=ground, aerial_image=aerial, ground_latlon=position,
                dx=dx, dy=dy, heading=heading, resolution=resolution,
                pose_variant=variant, pose_suffix=suffix,
                label_semantics='dataset camera position, not visible landmark',
                pixel_formula='x=W/2+dx; y=H/2-dy',
                aerial_GPS_present='aerial_latlon' in row,
                absolute_camera_survey_verified=False,
                pixels_checked=False, training_admitted=False)


def pose_increment_consistency(row):
    """Compare GPS DELTA to offset DELTA: internal consistency, not accuracy.

    Both variants use the same aerial reference, so its unknown GPS cancels.
    No aerial center is synthesized and no absolute residual is asserted.
    """
    raw = mapillary_pose(row, variant='raw')
    corrected = mapillary_pose(row, variant='opensfm')
    a, b = raw['ground_latlon'], corrected['ground_latlon']
    north = math.radians(b[0]-a[0]) * 6371008.8
    east = math.radians(b[1]-a[1]) * 6371008.8 * math.cos(math.radians((a[0]+b[0])/2))
    de = (corrected['dx']-raw['dx']) * raw['resolution']
    dn = (corrected['dy']-raw['dy']) * raw['resolution']
    residual = math.hypot(east-de, north-dn)
    return dict(gps_delta_east_m=east, gps_delta_north_m=north,
                offset_delta_east_m=de, offset_delta_north_m=dn,
                residual_m=residual, internally_agree_within1m=residual <= 1,
                absolute_camera_survey_verified=False,
                inferred_aerial_GPS=False)


def normalized_mapillary_rows(members, variant):
    rows = []
    for member in members:
        name = member['member']['name']
        if not name.startswith('OpenCVL_sample_data/mapillary/cities/') or not name.endswith('/labels.json'):
            raise ValueError('only acquired city sample metadata is admitted')
        for index, raw in enumerate(member['records']):
            p = mapillary_pose(raw, variant=variant)
            rows.append(dict(p, category='/'.join(name.split('/')[1:-1]),
                             label_member=name, row_index=index,
                             role='public_city_metadata_unassigned',
                             pixels_previously_acquired=False,
                             consistency=(pose_increment_consistency(raw)
                                          if raw.get('pose_mapilopensfm_available') is True
                                          else dict(available=False, reason='no corrected pose; raw remains explicit')),
                             training_eligibility=training_readiness(
                                 camera_point_label=True, geometry_checked=False,
                                 pixels_checked=False, role_authorized=False,
                                 split_isolated=False, profile_supported=False),
                             K_status=K_status(raw.get('K'))))
    return rows
