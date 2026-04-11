from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class JointLimit:
    name: str
    lower: float
    upper: float
    effort: float
    velocity: float

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) * 0.5

    @property
    def half_range(self) -> float:
        return max((self.upper - self.lower) * 0.5, 1e-6)


def parse_joint_limits(urdf_path: str | Path, joint_names: Iterable[str] | None = None) -> list[JointLimit]:
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    expected_names = set(joint_names or [])
    result: list[JointLimit] = []

    for joint_element in root.findall('joint'):
        joint_type = joint_element.attrib.get('type', '')
        if joint_type not in {'revolute', 'continuous'}:
            continue

        name = joint_element.attrib['name']
        if expected_names and name not in expected_names:
            continue

        limit_element = joint_element.find('limit')
        if limit_element is None:
            continue

        lower = float(limit_element.attrib.get('lower', '-3.14159'))
        upper = float(limit_element.attrib.get('upper', '3.14159'))
        effort = float(limit_element.attrib.get('effort', '0.0'))
        velocity = float(limit_element.attrib.get('velocity', '0.0'))
        result.append(
            JointLimit(
                name=name,
                lower=lower,
                upper=upper,
                effort=effort,
                velocity=velocity,
            )
        )

    if expected_names:
        found_names = {item.name for item in result}
        missing_names = expected_names - found_names
        if missing_names:
            missing_text = ', '.join(sorted(missing_names))
            raise ValueError(f'URDF 中缺少这些关节限制: {missing_text}')

    return result
