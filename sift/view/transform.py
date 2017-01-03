#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VisPy Transform objects to handle some of the more complex projections.

SIFT uses PROJ.4 to define geographic projections and these are rarely
possible to implement in Matrix transforms that come with VisPy.

"""

import re
from pyproj import Proj, pj_ellps
import numpy as np
from vispy.visuals.transforms import BaseTransform
from vispy.visuals.transforms._util import arg_to_vec4, as_vec4
from vispy.visuals.shaders import Function, Variable
from vispy.visuals.shaders.expression import TextExpression


class MacroExpression(TextExpression):
    macro_regex = re.compile(r'^#define\s+(?P<name>\w+)\s+(?P<expression>[^\s]+)')

    def __init__(self, text):
        match = self.macro_regex.match(text)
        if match is None:
            raise ValueError("Invalid macro definition: {}".format(text))
        match_dict = match.groupdict()
        self._name = match_dict['name']
        super(MacroExpression, self).__init__(text)

    def definition(self, names):
        return self.text

    @property
    def name(self):
        return self._name

COMMON_DEFINITIONS = """#define SPI     3.14159265359
#define TWOPI   6.2831853071795864769
#define ONEPI   3.14159265358979323846
#define M_PI    3.14159265358979310
#define M_PI_2  1.57079632679489660
#define M_PI_4  0.78539816339744828
#define M_FORTPI        M_PI_4                   /* pi/4 */
#define M_HALFPI        M_PI_2                   /* pi/2 */
#define M_PI_HALFPI     4.71238898038468985769   /* 1.5*pi */
#define M_TWOPI         6.28318530717958647693   /* 2*pi */
#define M_TWO_D_PI      M_2_PI                   /* 2/pi */
#define M_TWOPI_HALFPI  7.85398163397448309616   /* 2.5*pi */
"""
COMMON_DEFINITIONS = tuple(MacroExpression(line) for line in COMMON_DEFINITIONS.splitlines())

# proj_name -> (param_defaults, map_ellps, map_spher, imap_ellps, imap_spher)
# where 'map' is lon/lat to X/Y
# and 'imap' is X/Y to lon/lat
# WARNING: Need double {{ }} for functions for string formatting to work properly
PROJECTIONS = {
    'merc': (
        {'lon_0': 0.},
        """vec4 merc_map_e(vec4 pos) {{
            float lambda = radians(pos.x);
            float phi = radians(pos.y);
            float x = {a} * (lambda - {lon_0}f);
            float y = {a} * -log(pj_tsfn(phi, sin(phi), {e}));
            return vec4(x, y, pos.z, pos.w);
        }}""",
        """vec4 merc_map_s(vec4 pos) {{
            float lambda = radians(pos.x);
            {over}
            float phi = radians(pos.y);
            float x = {a} * (lambda - {lon_0}f);
            float y = {a} * log(tan(M_PI / 4.f + phi / 2.f));
            return vec4(x, y, pos.z, pos.w);
        }}""",
        """vec4 merc_imap_e(vec4 pos) {{
            float x = pos.x;
            float y = pos.y;
            float lambda = degrees({lon_0}f + x / {a});
            {over}
            float phi = degrees(pj_phi2(exp(-y / {a}), {e}));
            return vec4(lambda, phi, pos.z, pos.w);
        }}""",
        """vec4 merc_imap_s(vec4 pos) {{
            float x = pos.x;
            float y = pos.y;
            float lambda = degrees({lon_0}f + x / {a});
            {over}
            float phi = degrees(2.f * atan(exp(y / {a})) - M_PI / 2.f);
            return vec4(lambda, phi, pos.z, pos.w);
        }}""",
    ),
    # 'lcc': (),
}

# Misc GLSL functions used in one or more mapping functions above
adjlon_func = Function("""
    float adjlon(float lon) {
        if (abs(lon) <= M_PI) return (lon);
        lon += M_PI; // adjust to 0..2pi rad
        lon -= M_PI * 2 * floor(lon / M_PI / 2); // remove integral # of 'revolutions'
        lon -= M_PI;  // adjust back to -pi..pi rad
        return( lon );
    }
    """)

pj_msfn = Function("""
    float pj_msfn(float sinphi, float cosphi, float es) {
        return (cosphi / sqrt (1. - es * sinphi * sinphi));
    }
    """)

pj_tsfn = Function("""
    float pj_tsfn(float phi, float sinphi, float e) {
        sinphi *= e;
        return (tan (.5 * (M_HALFPI - phi)) /
           pow((1. - sinphi) / (1. + sinphi), .5 * e));
    }
    """)

pj_phi2 = Function("""
    float pj_phi2(float ts, float e) {
        float eccnth, Phi, con, dphi;

        eccnth = .5 * e;
        Phi = M_HALFPI - 2. * atan (ts);
        for (int i=15; i >= 0; --i) {
            con = e * sin(Phi);
            dphi = M_HALFPI - 2. * atan(ts * pow((1. - con) / (1. + con), eccnth)) - Phi;
            Phi += dphi;
            if (abs(dphi) <= 1.0e-10) {
                break;
            }
        }
        //if (i <= 0)
        //    pj_ctx_set_errno( ctx, -18 );
        return Phi;
    }
    """)


class PROJ4Transform(BaseTransform):
    glsl_map = None

    glsl_imap = None

    # Flags used to describe the transformation. Subclasses should define each
    # as True or False.
    # (usually used for making optimization decisions)

    # If True, then for any 3 colinear points, the
    # transformed points will also be colinear.
    Linear = False

    # The transformation's effect on one axis is independent
    # of the input position along any other axis.
    Orthogonal = False

    # If True, then the distance between two points is the
    # same as the distance between the transformed points.
    NonScaling = False

    # Scale factors are applied equally to all axes.
    Isometric = False

    def __init__(self, proj4_str, inverse=False):
        self.proj4_str = proj4_str
        self.proj = Proj(proj4_str)
        self._proj4_inverse = inverse
        proj_dict = self._proj_dict(proj4_str)

        # Get the specific functions for this projection
        proj_funcs = PROJECTIONS[proj_dict['proj']]
        # set default function parameters
        proj_args = proj_funcs[0].copy()
        proj_args.update(proj_dict)

        if proj_args.get('over'):
            proj_args['over'] = 'lambda = adjlon(lambda);'
        else:
            proj_args['over'] = ''

        if proj_dict['a'] == proj_dict['b']:
            # spheroid
            self.glsl_map = proj_funcs[2].format(**proj_args)
            self.glsl_imap = proj_funcs[4].format(**proj_args)
        else:
            # ellipsoid
            self.glsl_map = proj_funcs[1].format(**proj_args)
            self.glsl_imap = proj_funcs[3].format(**proj_args)

        if self._proj4_inverse:
            self.glsl_map, self.glsl_imap = self.glsl_imap, self.glsl_map

        super(PROJ4Transform, self).__init__()

        # Add common definitions and functions
        for d in COMMON_DEFINITIONS + (pj_tsfn, pj_phi2):
            self._shader_map._add_dep(d)
            self._shader_imap._add_dep(d)

        if proj_args['over']:
            self._shader_map._add_dep(adjlon_func)
            self._shader_imap._add_dep(adjlon_func)

        # print(self._shader_map.compile())

    def _proj_dict(self, proj_str):
        d = tuple(x.replace("+", "").split("=") for x in proj_str.split(" "))
        d = dict((x[0], x[1] if len(x) > 1 else 'true') for x in d)

        # convert numerical parameters to floats
        for k in d.keys():
            try:
                d[k] = float(d[k])
            except ValueError:
                pass

        # if they haven't provided a radius then they must have provided a datum or ellps
        if 'a' not in d:
            if 'datum' not in d:
                d.setdefault('ellps', d.setdefault('datum', 'WGS84'))
            else:
                d.setdefault('ellps', d.get('datum'))

        # if they provided an ellps/datum fill in information we know about it
        if d.get('ellps') is not None:
            # get information on the ellps being used
            ellps_info = pj_ellps[d['ellps']]
            for k in ['a', 'b', 'rf']:
                if k in ellps_info:
                    d.setdefault(k, ellps_info[k])

        # derive b, es, f, e
        if 'rf' not in d:
            if 'f' in d:
                d['rf'] = 1. / d['f']
            else:
                d['rf'] = d['a'] / (d['a'] - d['b'])
        if 'f' not in d:
            d['f'] = 1. / d['rf']
        if 'b' not in d:
            # a and rf must be in the dict
            d['b'] = d['a'] * (1. - d['f'])
        if 'es' not in d:
            if 'e' in d:
                d['es'] = d['e']**2
            else:
                d['es'] = 2 * d['f'] - d['f']**2
        if 'e' not in d:
            d['e'] = d['es']**0.5

        return d

    @arg_to_vec4
    def map(self, coords):
        """Map coordinates

        Parameters
        ----------
        coords : array-like
            Coordinates to map.
        """
        m = np.empty(coords.shape)
        m[:, 0], m[:, 1] = self.proj(coords[:, 0], coords[:, 1], inverse=self._proj4_inverse)
        m[:, 2:] = coords[:, 2:]
        return m

    @arg_to_vec4
    def imap(self, coords):
        """Inverse map coordinates

        Parameters
        ----------
        coords : array-like
            Coordinates to inverse map.
        """
        m = np.empty(coords.shape)
        m[:, 0], m[:, 1] = self.proj(coords[:, 0], coords[:, 1], inverse=not self._proj4_inverse)
        m[:, 2:] = coords[:, 2:]
        return m

    def __repr__(self):
        return "<%s:%s at 0x%x>" % (self.__class__.__name__, self.proj4_str, id(self))
