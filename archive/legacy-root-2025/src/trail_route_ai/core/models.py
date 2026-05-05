from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Set
from enum import Enum

@dataclass
class TrailSegment:
    seg_id: str
    name: str
    coordinates: List[Tuple[float, float]]
    length_ft: float
    direction: str
    required: bool = False
    access_from: Optional[str] = None # For parking info
    surface: str = 'unknown'
    exposure: str = 'mixed'  # full_sun, shaded, mixed
    avg_elevation: float = 0.0
    seasonal_closure: bool = False

    # The continuous planner uses these for graph compatibility
    start_node: Tuple[float, float] = field(init=False)
    end_node: Tuple[float, float] = field(init=False)

    def __post_init__(self):
        if self.coordinates:
            object.__setattr__(self, 'start_node', self.coordinates[0])
            object.__setattr__(self, 'end_node', self.coordinates[-1])
        else:
            object.__setattr__(self, 'start_node', None)
            object.__setattr__(self, 'end_node', None)

    @property
    def length_mi(self) -> float:
        return self.length_ft / 5280

    def is_closed_on(self, date) -> bool:
        """Check if segment is closed on given date"""
        # Simplified seasonal closure logic
        if self.seasonal_closure and self.avg_elevation > 6000:
            return date.month in [12, 1, 2, 3]  # Winter months
        return False

@dataclass
class Trailhead:
    """Represents a parking area and access point to trail systems"""
    name: str
    parking_coords: Tuple[float, float]  # (lat, lon)
    capacity: int
    access_type: str  # 'paved', 'gravel', '4wd'
    parking_type: str  # 'paved_lot', 'gravel_lot', 'roadside', 'informal'
    parking_fee: float = 0.0
    accessible_segments: Set[str] = field(default_factory=set)
    parking_notes: str = ""
    manually_verified: bool = False

@dataclass
class NavigationPoint:
    """A turn-by-turn navigation instruction"""
    distance_from_start: float  # miles
    instruction: str
    landmark: str = ""
    gps_coords: Optional[Tuple[float, float]] = None

@dataclass
class EscapeRoute:
    """An alternative route for bailing out of a long hike"""
    at_mile: float
    description: str
    saves_miles: float
    to_parking: str

@dataclass
class DrivingRoute:
    """Driving directions between locations"""
    distance: float  # miles
    time_minutes: int
    steps: List[str] = field(default_factory=list)

@dataclass
class Hike:
    """Represents a single continuous loop from a trailhead"""
    hike_number: int
    trailhead: str
    segments: List[TrailSegment] = field(default_factory=list)
    total_distance: float = 0.0
    elevation_gain: float = 0.0
    difficulty: str = 'moderate'  # easy, moderate, difficult, expert
    trail_conditions: str = ""
    estimated_minutes: int = 0
    
    # Navigation
    navigation_points: List[NavigationPoint] = field(default_factory=list)
    escape_routes: List[EscapeRoute] = field(default_factory=list)
    gpx_waypoints: List[Tuple[float, float]] = field(default_factory=list)
    
    # Parking and logistics
    parking_coords: Optional[Tuple[float, float]] = None
    parking_type: str = ""
    parking_fee: float = 0.0
    parking_notes: str = ""
    
    # Driving to next location
    driving_directions: Optional[List[str]] = None
    drive_time_minutes: int = 0
    drive_distance_miles: float = 0.0
    drive_to_distance: float = 0.0  # alias for compatibility
    
    # Timing recommendations
    recommended_start_time: Optional[str] = None
    includes_snow_warning: bool = False
    max_elevation: float = 0.0
    
    @property
    def id(self) -> str:
        return f"hike_{self.hike_number}"
    
    @property
    def total_distance_mi(self) -> float:
        """Alias for compatibility"""
        return self.total_distance

@dataclass
class Day:
    """Represents a single day's hiking plan"""
    number: int
    hikes: List[Hike] = field(default_factory=list)
    total_distance: float = 0.0
    total_elevation: float = 0.0
    total_driving: float = 0.0
    type: str = 'medium'  # short, medium, long
    
    # Weather contingencies
    hot_weather_alternative: Optional[str] = None
    storm_bailout_points: Optional[List[str]] = None
    
    @property
    def is_weekend(self) -> bool:
        # Simplified - assume weekends based on day number
        return (self.number % 7) in [6, 0]

@dataclass 
class TrailFamily:
    """A group of related trail segments that form a natural system"""
    name: str
    segments: List[TrailSegment] = field(default_factory=list)
    connector_segments: List[TrailSegment] = field(default_factory=list)
    required_segments: List[TrailSegment] = field(default_factory=list)

@dataclass
class Loop:
    """A closed loop route covering segments efficiently"""
    trailhead: str
    segments: List[TrailSegment] = field(default_factory=list)
    total_distance: float = 0.0
    required_coverage: Set[str] = field(default_factory=set)
    connector_ratio: float = 0.0

@dataclass
class GeneratedPlan:
    """Complete plan covering all required segments"""
    days: List[Day] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def all_hikes(self) -> List[Hike]:
        return [hike for day in self.days for hike in day.hikes]

@dataclass
class PlannerConfig:
    """Loads and holds the configuration for the trailhead-based planner."""
    solver_time_limit_seconds: int
    trailhead_depots: List[Dict[str, Any]]
    daily_capacities: Dict[str, Dict[str, int]]
    cost_model: Dict[str, float]
    drive_threshold_miles: float 
    base_pace_min_per_mile: float = 16.0
    short_day_limit: float = 6.0
    medium_day_limit: float = 15.0
    long_day_limit: float = 25.0 