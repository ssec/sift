#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VisPy Transform objects to handle some of the more complex projections.

SIFT uses PROJ.4 to define geographic projections and these are rarely
possible to implement in Matrix transforms that come with VisPy.

"""

import re
from os import linesep as os_linesep
from typing import List

import numpy as np
from pyproj import Proj, pj_ellps
from vispy import glsl
from vispy.visuals.shaders import Function
from vispy.visuals.shaders.expression import TextExpression
from vispy.visuals.shaders.parsing import find_program_variables
from vispy.visuals.transforms import BaseTransform
from vispy.visuals.transforms._util import arg_to_vec4


class VariableDeclaration(TextExpression):
    """TextExpression subclass for exposing GLSL variables to vispy glsl interface.

    Parameters
    ----------
    name : str
        Name of the variable.
    text : str
        Rvalue to be assigned to the variable.
    """

    def __init__(self, name: str, text: str) -> None:
        self._name = name
        super().__init__(text)

    def definition(self, names, version=None, shader=None) -> str:
        return self.text

    @property
    def name(self) -> str:
        return self._name


class GLSL_Adapter(TextExpression):
    """TextExpression subclass for parsing Macro definitions from .glsl header files and exposing them to vispy.

    This class makes macro definitions accessible to vispy's shader code
    processing. Assumes .glsl code to be parsed is accessible as python string. For reading
    .glsl header code from a file see GLSL_FileAdapter subclass.

    Parameters
    ----------
    text : str
        Actual .glsl code string.
    """

    _expr_list: list = []

    def __init__(self, text: str) -> None:
        # Regular expression for parsing possibly include-guarded macro definitions from .glsl
        # header files; makes strong assumptions about formatting of macro names by assuming
        # underscores in front of and behind the macro name. In line with vispy .glsl shader code.
        guard_pattern = re.compile(r"^#ifndef\s*(?P<guard>_[A-Za-z]*_)")
        _guard_flag = False
        for line in text.splitlines():
            match_guard = guard_pattern.match(line)
            var_match = find_program_variables(line)
            if match_guard is not None:
                _name = match_guard["guard"]
                _text = match_guard.group(0) + os_linesep + f"#define {_name}"
                self._expr_list.append(VariableDeclaration(_name, _text))
                self._expr_list.append(VariableDeclaration(_name + "EIF", "#endif"))
                _guard_flag = True
            elif var_match is not None:
                key_list = list(var_match.keys())
                if len(key_list) > 1:
                    raise ValueError("More than one variable definition per line " "not supported.")
                elif len(key_list) != 0:
                    self._expr_list.append(VariableDeclaration(key_list[0], line))
        if _guard_flag:
            # in case of include guards, shift #endif to bottom of
            # expression list to match #ifndef
            eif_token = self._expr_list[1]
            self._expr_list[1:-1] = self._expr_list[2:]
            self._expr_list[-1] = eif_token

    @property
    def expr_list(self) -> List[VariableDeclaration]:
        return self._expr_list


class GLSL_FileAdapter(GLSL_Adapter):
    """GLSL_Adapter subclass adding the functionality to read .glsl header code from files.

    Parameters
    ----------
    file_path : str
        Path to .glsl header file.
    """

    def __init__(self, file_path: str) -> None:
        text = glsl.get(file_path)
        super(GLSL_FileAdapter, self).__init__(text)


COMMON_VALUES_DEF = """const float SPI = 3.14159265359;
const float TWOPI = 6.2831853071795864769;
const float ONEPI = 3.14159265358979323846;
const float M_FORTPI = M_PI_4;                      /* pi/4 */
const float M_HALFPI = M_PI_2;                      /* pi/2 */
const float M_PI_HALFPI = 4.71238898038468985769;   /* 1.5*pi */
const float M_TWOPI = 6.28318530717958647693;       /* 2*pi */
const float M_TWO_D_PI = 2.0/M_PI;                  /* 2/pi */
const float M_TWOPI_HALFPI = 2.5 / M_PI;            /* 2.5*pi */
"""

math_consts = GLSL_FileAdapter("math/constants.glsl").expr_list
COMMON_VALUES = GLSL_Adapter(COMMON_VALUES_DEF).expr_list
M_FORTPI = M_PI_4 = 0.78539816339744828
M_HALFPI = M_PI_2 = 1.57079632679489660


def merc_init(proj_dict):
    proj_dict.setdefault("lon_0", 0.0)
    proj_dict.setdefault("k0", 1.0)

    phits = 0.0
    is_phits = "lat_ts" in proj_dict
    if is_phits:
        phits = np.radians(proj_dict["lat_ts"])
        if phits >= M_HALFPI:
            raise ValueError("PROJ.4 'lat_ts' parameter must be greater than PI/2")

    if proj_dict["a"] != proj_dict["b"]:
        # ellipsoid
        if is_phits:
            proj_dict["k0"] = pj_msfn_py(np.sin(phits), np.cos(phits), proj_dict["es"])
    elif is_phits:
        # spheroid
        proj_dict["k0"] = np.cos(phits)

    return proj_dict


def lcc_init(proj_dict):
    if "lat_1" not in proj_dict:
        raise ValueError("PROJ.4 'lat_1' parameter is required for 'lcc' projection")

    proj_dict.setdefault("lon_0", 0.0)
    if "lat_2" not in proj_dict:
        proj_dict["lat_2"] = proj_dict["lat_1"]
        if "lat_0" not in proj_dict:
            proj_dict["lat_0"] = proj_dict["lat_1"]
    proj_dict["phi1"] = np.radians(proj_dict["lat_1"])
    proj_dict["phi2"] = np.radians(proj_dict["lat_2"])
    proj_dict["phi0"] = np.radians(proj_dict["lat_0"])

    if abs(proj_dict["phi1"] + proj_dict["phi2"]) < 1e-10:
        raise ValueError("'lat_1' + 'lat_2' for 'lcc' projection when converted to radians must be greater than 1e-10.")

    proj_dict["n"] = sinphi = np.sin(proj_dict["phi1"])
    cosphi = np.cos(proj_dict["phi1"])
    secant = abs(proj_dict["phi1"] - proj_dict["phi2"]) >= 1e-10
    proj_dict["ellips"] = proj_dict["a"] != proj_dict["b"]
    if proj_dict["ellips"]:
        # ellipsoid
        m1 = pj_msfn_py(sinphi, cosphi, proj_dict["es"])
        ml1 = pj_tsfn_py(proj_dict["phi1"], sinphi, proj_dict["e"])
        if secant:
            sinphi = np.sin(proj_dict["phi2"])
            proj_dict["n"] = np.log(m1 / pj_msfn_py(sinphi, np.cos(proj_dict["phi2"]), proj_dict["es"]))
            proj_dict["n"] /= np.log(ml1 / pj_tsfn_py(proj_dict["phi2"], sinphi, proj_dict["e"]))
        proj_dict["c"] = proj_dict["rho0"] = m1 * pow(ml1, -proj_dict["n"]) / proj_dict["n"]
        proj_dict["rho0"] *= (
            0.0
            if abs(abs(proj_dict["phi0"]) - M_HALFPI) < 1e-10
            else pow(pj_tsfn_py(proj_dict["phi0"], np.sin(proj_dict["phi0"]), proj_dict["e"]), proj_dict["n"])
        )
    else:
        # spheroid
        if secant:
            proj_dict["n"] = np.log(cosphi / np.cos(proj_dict["phi2"])) / np.log(
                np.tan(M_FORTPI + 0.5 * proj_dict["phi2"]) / np.tan(M_FORTPI + 0.5 * proj_dict["phi1"])
            )
        proj_dict["c"] = cosphi * pow(np.tan(M_FORTPI + 0.5 * proj_dict["phi1"]), proj_dict["n"]) / proj_dict["n"]
        proj_dict["rho0"] = (
            0.0
            if abs(abs(proj_dict["phi0"]) - M_HALFPI) < 1e-10
            else proj_dict["c"] * pow(np.tan(M_FORTPI + 0.5 * proj_dict["phi0"]), -proj_dict["n"])
        )
    proj_dict["ellips"] = "true" if proj_dict["ellips"] else "false"
    return proj_dict


def geos_init(proj_dict):
    if "h" not in proj_dict:
        raise ValueError("PROJ.4 'h' parameter is required for 'geos' projection")

    proj_dict.setdefault("lat_0", 0.0)
    proj_dict.setdefault("lon_0", 0.0)
    # lat_0 is set to phi0 in the PROJ.4 C source code
    # if 'lat_0' not in proj_dict:
    #     raise ValueError("PROJ.4 'lat_0' parameter is required for 'geos' projection")

    if "sweep" not in proj_dict or proj_dict["sweep"] is None:
        proj_dict["flip_axis"] = "false"
    elif proj_dict["sweep"] not in ["x", "y"]:
        raise ValueError("PROJ.4 'sweep' parameter must be 'x' or 'y'")
    elif proj_dict["sweep"] == "x":
        proj_dict["flip_axis"] = "true"
    else:
        proj_dict["flip_axis"] = "false"

    proj_dict["radius_g_1"] = proj_dict["h"] / proj_dict["a"]
    proj_dict["radius_g"] = 1.0 + proj_dict["radius_g_1"]
    proj_dict["C"] = proj_dict["radius_g"] * proj_dict["radius_g"] - 1.0
    if proj_dict["a"] != proj_dict["b"]:
        # ellipsoid
        proj_dict["one_es"] = 1.0 - proj_dict["es"]
        proj_dict["rone_es"] = 1.0 / proj_dict["one_es"]
        proj_dict["radius_p"] = np.sqrt(proj_dict["one_es"])
        proj_dict["radius_p2"] = proj_dict["one_es"]
        proj_dict["radius_p_inv2"] = proj_dict["rone_es"]
    else:
        proj_dict["radius_p"] = proj_dict["radius_p2"] = proj_dict["radius_p_inv2"] = 1.0
    return proj_dict


def stere_init(proj_dict):
    # Calculate phits
    phits = abs(np.radians(proj_dict["lat_ts"]) if "lat_ts" in proj_dict else M_HALFPI)
    # Determine mode
    if abs(abs(np.radians(proj_dict["lat_0"])) - M_HALFPI) < 1e-10:
        # Assign "mode" in proj_dict to be GLSL for specific case (make sure to handle C-code case fallthrough):
        # 0 = n_pole, 1 = s_pole.
        proj_dict["mode"] = 1 if proj_dict["lat_0"] < 0 else 0
        if proj_dict["a"] != proj_dict["b"]:
            # ellipsoid
            e = proj_dict["e"]
            if abs(phits - M_HALFPI) < 1e-10:
                proj_dict["akm1"] = 2.0 / np.sqrt((1 + e) ** (1 + e) * (1 - e) ** (1 - e))
            else:
                proj_dict["akm1"] = np.cos(phits) / (
                    pj_tsfn_py(phits, np.sin(phits), e) * np.sqrt(1.0 - (np.sin(phits) * e) ** 2)
                )
        else:
            # sphere
            proj_dict["akm1"] = (
                np.cos(phits) / np.tan(M_FORTPI - 0.5 * phits) if abs(phits - M_HALFPI) >= 1e-10 else 2.0
            )
    else:
        # If EQUIT or OBLIQ mode:
        raise NotImplementedError("This projection mode is not supported yet.")
    return proj_dict


def eqc_init(proj_dict):
    proj_dict.setdefault("lat_0", 0.0)
    proj_dict.setdefault("lat_ts", proj_dict["lat_0"])
    proj_dict["rc"] = np.cos(np.radians(proj_dict["lat_ts"]))
    if proj_dict["rc"] <= 0.0:
        raise ValueError("PROJ.4 'lat_ts' parameter must be in range (-PI/2,PI/2)")
    proj_dict["phi0"] = np.radians(proj_dict["lat_0"])
    proj_dict["es"] = 0.0
    return proj_dict


def latlong_init(proj_dict):
    if "over" in proj_dict:
        # proj_dict['offset'] = '360.'
        proj_dict["offset"] = "0."
    else:
        proj_dict["offset"] = "0."
    return proj_dict


# proj_name -> (proj_init, map_ellps, map_spher, imap_ellps, imap_spher)
# where 'map' is lon/lat to X/Y
# and 'imap' is X/Y to lon/lat
# WARNING: Need double {{ }} for functions for string formatting to work properly
PROJECTIONS = {
    "longlat": (
        latlong_init,
        """vec4 latlong_map(vec4 pos) {{
            return vec4(pos.x + {offset}, pos.y, pos.z, pos.w);
        }}""",
        """vec4 latlong_map(vec4 pos) {{
            return vec4(pos.x + {offset}, pos.y, pos.z, pos.w);
        }}""",
        """vec4 latlong_imap(vec4 pos) {{
            return pos;
        }}""",
        """vec4 latlong_imap(vec4 pos) {{
            return pos;
        }}""",
    ),
    "merc": (
        merc_init,
        """vec4 merc_map_e(vec4 pos) {{
            float lambda = radians(pos.x);
            {over}
            float phi = radians(pos.y);
            if (abs(abs(phi) - M_HALFPI) <= 1.e-10) {{
                return vec4(1. / 0., 1. / 0., pos.z, pos.w);
            }}
            float x = {a} * {k0} * (lambda - {lon_0}f);
            float y = {a} * {k0} * -log(pj_tsfn(phi, sin(phi), {e}));
            return vec4(x, y, pos.z, pos.w);
        }}""",
        """vec4 merc_map_s(vec4 pos) {{
            float lambda = radians(pos.x);
            {over}
            float phi = radians(pos.y);
            if (abs(abs(phi) - M_HALFPI) <= 1.e-10) {{
                return vec4(1. / 0., 1. / 0., pos.z, pos.w);
            }}
            float x = {a} * {k0} * (lambda - {lon_0}f);
            float y = {a} * {k0} * log(tan(M_PI / 4.f + phi / 2.f));
            return vec4(x, y, pos.z, pos.w);
        }}""",
        """vec4 merc_imap_e(vec4 pos) {{
            float x = pos.x;
            float y = pos.y;
            float lambda = {lon_0}f + x / ({a} * {k0});
            {over}
            lambda = degrees(lambda);
            float phi = degrees(pj_phi2(exp(-y / ({a} * {k0})), {e}));
            return vec4(lambda, phi, pos.z, pos.w);
        }}""",
        """vec4 merc_imap_s(vec4 pos) {{
            float x = pos.x;
            float y = pos.y;
            float lambda = {lon_0}f + x / ({a} * {k0});
            {over}
            lambda = degrees(lambda);
            float phi = degrees(2.f * atan(exp(y / ({a} * {k0}))) - M_PI / 2.f);
            return vec4(lambda, phi, pos.z, pos.w);
        }}""",
    ),
    "lcc": (
        lcc_init,
        """vec4 lcc_map_e(vec4 pos) {{
            float rho;
            float lambda = radians(pos.x - {lon_0});
            float phi = radians(pos.y);
            {over}

            if (abs(abs(phi) - M_HALFPI) < 1e-10) {{
                if ((phi * {n}) <= 0.) {{
                    return vec4(1. / 0., 1. / 0., pos.z, pos.w);
                }}
                rho = 0.;
            }} else {{
                rho = {c} * ({ellips} ? pow(pj_tsfn(phi, sin(phi),
                    {e}), {n}) : pow(tan(M_FORTPI + .5 * phi), -1. * {n}));
            }}

            lambda *= {n};
            return vec4({a} * (rho * sin(lambda)), {a} * ({rho0} - rho * cos(lambda)), pos.z, pos.w);
        }}""",
        None,
        """vec4 lcc_imap_e(vec4 pos) {{
            float rho, phi, lambda;
            float x = pos.x / {a};
            float y = pos.y / {a};
            y = {rho0} - y;
            rho = hypot(x, y);
            if (rho != 0.0) {{
                if ({n} < 0.) {{
                    rho = -rho;
                    x = -x;
                    y = -y;
                }}
                if ({ellips}) {{
                    phi = pj_phi2(pow(rho / {c}, 1. / {n}), {e});
                    //if (phi == HUGE_VAL) {{
                    //    return vec4(1. / 0., 1. / 0., pos.z, pos.w);
                    //}}
                }} else {{
                    phi = 2. * atan(pow({c} / rho, 1. / {n})) - M_HALFPI;
                }}
                // atan2 in C
                lambda = atan(x, y) / {n};
            }} else {{
                lambda = 0.;
                phi = {n} > 0. ? M_HALFPI : - M_HALFPI;
            }}
            {over}
            return vec4(degrees(lambda) + {lon_0}, degrees(phi), pos.z, pos.w);
        }}""",
        None,
    ),
    "geos": (
        geos_init,
        """vec4 geos_map_e(vec4 pos) {{
            float lambda, phi, r, Vx, Vy, Vz, tmp, x, y;
            lambda = radians(pos.x - {lon_0});
            {over}
            phi = atan({radius_p2} * tan(radians(pos.y)));
            r = {radius_p} / hypot({radius_p} * cos(phi), sin(phi));
            Vx = r * cos(lambda) * cos(phi);
            Vy = r * sin(lambda) * cos(phi);
            Vz = r * sin(phi);

            // TODO: Best way to 'discard' a vertex
            if ((({radius_g} - Vx) * Vx - Vy * Vy - Vz * Vz * {radius_p_inv2}) < 0.) {{
               return vec4(1. / 0., 1. / 0., pos.z, pos.w);
            }}

            tmp = {radius_g} - Vx;

            if ({flip_axis}) {{
                x = {radius_g_1} * atan(Vy / hypot(Vz, tmp));
                y = {radius_g_1} * atan(Vz / tmp);
            }} else {{
                x = {radius_g_1} * atan(Vy / tmp);
                y = {radius_g_1} * atan(Vz / hypot(Vy, tmp));
            }}
            return vec4(x * {a}, y * {a}, pos.z, pos.w);
        }}""",
        """vec4 geos_map_s(vec4 pos) {{
            float lambda, phi, Vx, Vy, Vz, tmp, x, y;
            lambda = radians(pos.x - {lon_0});
            {over}
            phi = radians(pos.y);
            Vx = cos(lambda) * cos(phi);
            Vy = sin(lambda) * cos(phi);
            Vz = sin(phi);
            // TODO: Best way to 'discard' a vertex
            if ((({radius_g} - Vx) * Vx - Vy * Vy - Vz * Vz * {radius_p_inv2}) < 0.) {{
               return vec4(1. / 0., 1. / 0., pos.z, pos.w);
            }}
            tmp = {radius_g} - Vx;
            if ({flip_axis}) {{
                x = {a} * {radius_g_1} * atan(Vy / hypot(Vz, tmp));
                y = {a} * {radius_g_1} * atan(Vz / tmp);
            }}
            else {{
                x = {a} * {radius_g_1} * atan(Vy / tmp);
                y = {a} * {radius_g_1} * atan(Vz / hypot(Vy, tmp));
            }}
            return vec4(x, y, pos.z, pos.w);
        }}""",
        """vec4 geos_imap_e(vec4 pos) {{
            float a, b, k, det, x, y, Vx, Vy, Vz, lambda, phi;
            x = pos.x / {a};
            y = pos.y / {a};

            Vx = -1.0;
            if ({flip_axis}) {{
                Vz = tan(y / {radius_g_1});
                Vy = tan(x / {radius_g_1}) * hypot(1.0, Vz);
            }} else {{
                Vy = tan(x / {radius_g_1});
                Vz = tan(y / {radius_g_1}) * hypot(1.0, Vy);
            }}

            a = Vz / {radius_p};
            a = Vy * Vy + a * a + Vx * Vx;
            b = 2 * {radius_g} * Vx;
            det = ((b * b) - 4 * a * {C});
            if (det < 0.) {{
                // FIXME
                return vec4(1. / 0., 1. / 0., pos.z, pos.w);
            }}

            k = (-b - sqrt(det)) / (2. * a);
            Vx = {radius_g} + k * Vx;
            Vy *= k;
            Vz *= k;

            // atan2 in C
            lambda = atan(Vy, Vx);
            {over}
            phi = atan(Vz * cos(lambda) / Vx);
            phi = atan({radius_p_inv2} * tan(phi));
            return vec4(degrees(lambda) + {lon_0}, degrees(phi), pos.z, pos.w);
        }}""",
        """vec4 geos_imap_s(vec4 pos) {{
            float x, y, Vx, Vy, Vz, a, b, k, det, lambda, phi;
            x = pos.x / {a};
            y = pos.y / {a};
            Vx = -1.;
            if ({flip_axis}) {{
                Vz = tan(y / ({radius_g} - 1.));
                Vy = tan(x / ({radius_g} - 1.)) * sqrt(1. + Vz * Vz);
            }}
            else {{
                Vy = tan(x / ({radius_g} - 1.));
                Vz = tan(y / ({radius_g} - 1.)) * sqrt(1. + Vy * Vy);
            }}
            a = Vy * Vy + Vz * Vz + Vx * Vx;
            b = 2 * {radius_g} * Vx;
            det = b * b - 4 * a * {C};
            if (det < 0.) {{
                return vec4(1. / 0., 1. / 0., pos.z, pos.w);
            }}
            k = (-b - sqrt(det)) / (2 * a);
            Vx = {radius_g} + k * Vx;
            Vy *= k;
            Vz *= k;
            lambda = atan(Vy, Vx);
            {over}
            phi = atan(Vz * cos(lambda) / Vx);
            return vec4(degrees(lambda) + {lon_0}, degrees(phi), pos.z, pos.w);
        }}""",
    ),
    "stere": (
        stere_init,
        """vec4 stere_map_e(vec4 pos) {{
            float lambda, phi, coslam, sinlam, sinphi, x, y;
            lambda = radians(pos.x - {lon_0});
            {over}
            phi = radians(pos.y);
            coslam = cos(lambda);
            sinlam = sin(lambda);
            sinphi = sin(phi);
            if ({mode} == 1) {{
                phi = -phi;
                coslam = - coslam;
                sinphi = -sinphi;
            }}
            x = {akm1} * pj_tsfn(phi, sinphi, {e});
            y = {a} * -x * coslam;
            x *= {a} * sinlam;
            return vec4(x, y, pos.z, pos.w);
        }}""",
        """vec4 stere_map_s(vec4 pos) {{
            float lambda, phi, coslam, sinlam, x, y;
            lambda = radians(pos.x - {lon_0});
            {over}
            phi = radians(pos.y);
            coslam = cos(lambda);
            sinlam = sin(lambda);
            if ({mode} == 0) {{
                coslam = - coslam;
                phi = - phi;
            }}
            if (abs(phi - M_HALFPI) < 1.e-8) {{
                return vec4(1. / 0., 1. / 0., pos.z, pos.w);
            }}
            y = {akm1} * tan(M_FORTPI + .5 * phi);
            x = {a} * sinlam * y;
            y *= {a} * coslam;
            return vec4(x, y, pos.z, pos.w);
        }}""",
        """vec4 stere_imap_e(vec4 pos) {{
            float x, y, phi, lambda, tp, phi_l, sinphi;
            x = pos.x / {a};
            y = pos.y / {a};
            phi = radians(y);
            lambda = radians(x);
            tp = -hypot(x,y) / {akm1};
            phi_l = M_HALFPI - 2. * atan(tp);
            sinphi = 0.;
            if ({mode} == 0) {{
                y = -y;
            }}
            for (int i = 8; i-- > 0; phi_l = phi) {{
                sinphi = {e} * sin(phi_l);
                phi = 2. * atan(tp * pow((1. + sinphi) / (1. - sinphi), -.5 * {e})) + M_HALFPI;
                if (abs(phi_l - phi) < 1.e-10) {{
                    if ({mode} == 1) {{
                        phi = -phi;
                    }}
                    lambda = (x == 0. && y == 0.) ? 0. : atan(x, y);
                    {over}
                    return vec4(degrees(lambda) + {lon_0}, degrees(phi), pos.z, pos.w);
                }}
            }}
            return vec4(1. / 0., 1. / 0., pos.z, pos.w);
        }}""",
        """vec4 stere_imap_s(vec4 pos) {{
            float x, y, rh, cosc, phi, lambda;
            x = pos.x / {a};
            y = pos.y / {a};
            rh = hypot(x, y);
            cosc = cos(2. * atan(rh / {akm1}));
            phi = 0;
            if ({mode} == 0) {{
                y = -y;
            }}
            if (abs(rh) < 1.e-10) {{
                phi = radians({lat_0});
            }}
            else {{
                phi = asin({mode} == 1 ? -cosc : cosc);
            }}
            lambda = (x == 0. && y == 0.) ? 0. : atan(x, y);
            {over}
            return vec4(degrees(lambda) + {lon_0}, degrees(phi), pos.z, pos.w);
        }}""",
    ),
    "eqc": (
        eqc_init,
        None,
        """vec4 eqc_map_s(vec4 pos) {{
        {es};
        float lambda = radians(pos.x);
        {over}
        float phi = radians(pos.y);
        float x = {a} * {rc} * lambda;
        float y = {a} * (phi - {phi0});
        return vec4(x, y, pos.z, pos.w);
    }}""",
        None,
        """vec4 eqc_imap_s(vec4 pos) {{
            float x = pos.x / {a};
            float y = pos.y / {a};
            float lambda = x / {rc};
            {over}
            float phi = y + {phi0};
            return vec4(degrees(lambda), degrees(phi), pos.z, pos.w);
        }}""",
    ),
}
PROJECTIONS["lcc"] = (
    lcc_init,
    PROJECTIONS["lcc"][1],
    PROJECTIONS["lcc"][1],
    PROJECTIONS["lcc"][3],
    PROJECTIONS["lcc"][3],
)

# Misc GLSL functions used in one or more mapping functions above
adjlon_func = Function(
    """
    float adjlon(float lon) {
        if (abs(lon) <= M_PI) return (lon);
        lon += M_PI; // adjust to 0..2pi rad
        lon -= M_TWOPI * floor(lon / M_TWOPI); // remove integral # of 'revolutions'
        lon -= M_PI;  // adjust back to -pi..pi rad
        return( lon );
    }
    """
)

# handle prime meridian shifts
pm_func_str = """
    float adjlon(float lon) {{
        return lon + radians({pm});
    }}
"""

pj_msfn = Function(
    """
    float pj_msfn(float sinphi, float cosphi, float es) {
        return (cosphi / sqrt (1. - es * sinphi * sinphi));
    }
    """
)


def pj_msfn_py(sinphi, cosphi, es):
    return cosphi / np.sqrt(1.0 - es * sinphi * sinphi)


pj_tsfn = Function(
    """
    float pj_tsfn(float phi, float sinphi, float e) {
        sinphi *= e;
        return (tan (.5 * (M_HALFPI - phi)) /
           pow((1. - sinphi) / (1. + sinphi), .5 * e));
    }
    """
)


def pj_tsfn_py(phi, sinphi, e):
    sinphi *= e
    return np.tan(0.5 * (M_HALFPI - phi)) / pow((1.0 - sinphi) / (1.0 + sinphi), 0.5 * e)


pj_phi2 = Function(
    """
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
    """
)

hypot = Function(
    """
float hypot(float x, float y) {
    if ( x < 0.)
        x = -x;
    else if (x == 0.)
        return (y < 0. ? -y : y);
    if (y < 0.)
        y = -y;
    else if (y == 0.)
        return (x);
    if ( x < y ) {
        x /= y;
        return ( y * sqrt( 1. + x * x ) );
    } else {
        y /= x;
        return ( x * sqrt( 1. + y * y ) );
    }
}
"""
)


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
        proj_dict = self._create_proj_dict(proj4_str)

        # Get the specific functions for this projection
        proj_funcs = PROJECTIONS[proj_dict["proj"]]
        # set default function parameters
        proj_init = proj_funcs[0]
        proj_args = proj_init(proj_dict)

        if "pm" in proj_args:
            # force to float
            proj_args["pm"] = float(proj_args["pm"])
            proj_args["over"] = "lambda = adjlon(lambda);"
        elif proj_args.get("over"):
            proj_args["over"] = ""
        else:
            proj_args["over"] = "lambda = adjlon(lambda);"

        if proj_dict["a"] == proj_dict["b"]:
            # spheroid
            self.glsl_map = proj_funcs[2]
            self.glsl_imap = proj_funcs[4]
            if self.glsl_map is None or self.glsl_imap is None:
                raise ValueError("Spheroid transform for {} not implemented yet".format(proj_dict["proj"]))
        else:
            # ellipsoid
            self.glsl_map = proj_funcs[1]
            self.glsl_imap = proj_funcs[3]
            if self.glsl_map is None or self.glsl_imap is None:
                raise ValueError("Ellipsoid transform for {} not implemented yet".format(proj_dict["proj"]))

        self.glsl_map = self.glsl_map.format(**proj_args)
        self.glsl_imap = self.glsl_imap.format(**proj_args)

        if self._proj4_inverse:
            self.glsl_map, self.glsl_imap = self.glsl_imap, self.glsl_map

        super(PROJ4Transform, self).__init__()

        # Add common definitions and functions
        for d in math_consts + COMMON_VALUES + [pj_tsfn, pj_phi2, hypot]:
            self._shader_map._add_dep(d)
            self._shader_imap._add_dep(d)

        if "pm" in proj_args:
            pm_func = Function(pm_func_str.format(**proj_args))
            self._shader_map._add_dep(pm_func)
            self._shader_imap._add_dep(pm_func)
        elif proj_args["over"]:
            self._shader_map._add_dep(adjlon_func)
            self._shader_imap._add_dep(adjlon_func)

        # Add special handling of possible infinity lon/lat values
        self._shader_map[
            "pre"
        ] = """
    if (abs(pos.x) > 1e30 || abs(pos.y) > 1e30)
        return vec4(1. / 0., 1. / 0., pos.z, pos.w);
        """

        # print(self._shader_map.compile())

    @property
    def is_geographic(self):
        if hasattr(self.proj, "crs"):
            # pyproj 2.0+
            return self.proj.crs.is_geographic
        return self.proj.is_latlong()

    def _create_proj_dict(self, proj_str):  # noqa: C901
        d = tuple(x.replace("+", "").split("=") for x in proj_str.split(" "))
        d = dict((x[0], x[1] if len(x) > 1 else "true") for x in d)

        # convert numerical parameters to floats
        for k in d.keys():
            try:
                d[k] = float(d[k])
            except ValueError:
                pass

        d["proj4_str"] = proj_str

        # if they haven't provided a radius then they must have provided a datum or ellps
        if "R" in d:
            # spheroid
            d.setdefault("a", d["R"])
            d.setdefault("b", d["R"])
        if "a" not in d:
            if "datum" not in d:
                d.setdefault("ellps", d.setdefault("datum", "WGS84"))
            else:
                d.setdefault("ellps", d.get("datum"))

        # if they provided an ellps/datum fill in information we know about it
        if d.get("ellps") is not None:
            # get information on the ellps being used
            ellps_info = pj_ellps[d["ellps"]]
            for k in ["a", "b", "rf"]:
                if k in ellps_info:
                    d.setdefault(k, ellps_info[k])

        # derive b, es, f, e
        if "rf" not in d:
            if "f" in d:
                d["rf"] = 1.0 / d["f"]
            elif d["a"] == d["b"]:
                d["rf"] = 0.0
            else:
                d["rf"] = d["a"] / (d["a"] - d["b"])
        if "f" not in d:
            if d["rf"]:
                d["f"] = 1.0 / d["rf"]
            else:
                d["f"] = 0.0
        if "b" not in d:
            # a and rf must be in the dict
            d["b"] = d["a"] * (1.0 - d["f"])
        if "es" not in d:
            if "e" in d:
                d["es"] = d["e"] ** 2
            else:
                d["es"] = 2 * d["f"] - d["f"] ** 2
        if "e" not in d:
            d["e"] = d["es"] ** 0.5

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
        if self.is_geographic:
            m[:, 0] = coords[:, 0]
            m[:, 1] = coords[:, 1]
        else:
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
        if self.is_geographic:
            m[:, 0] = coords[:, 0]
            m[:, 1] = coords[:, 1]
        else:
            m[:, 0], m[:, 1] = self.proj(coords[:, 0], coords[:, 1], inverse=not self._proj4_inverse)
        m[:, 2:] = coords[:, 2:]
        return m

    def __repr__(self):
        return "<%s:%s at 0x%x>" % (self.__class__.__name__, self.proj4_str, id(self))
