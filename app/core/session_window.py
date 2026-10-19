from dataclasses import dataclass
from datetime import datetime, date, time, timedelta, timezone
from zoneinfo import ZoneInfo
from app.core.enums import SessionType
from app.core.constants import FUTURE_CLOCK_SKEW_MINUTES

EASTERN_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class SessionWindow:
    market_timezone: str
    session_type: SessionType
    is_market_open: bool
    window_start_utc: datetime
    window_start_et: datetime
    window_end_utc: datetime
    window_end_et: datetime
    window_duration_hours: float
    run_timestamp_utc: datetime
    run_timestamp_et: datetime

    def is_timestamp_in_window(self, dt_utc: datetime | None) -> bool:
        """
        Check if an article's UTC timestamp falls within this session window,
        allowing at most FUTURE_CLOCK_SKEW_MINUTES minutes of future clock skew. Returns False if timestamp is None.
        """
        if dt_utc is None:
            return False
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        skew_allowance = timedelta(minutes=FUTURE_CLOCK_SKEW_MINUTES)
        return self.window_start_utc <= dt_utc <= (self.window_end_utc + skew_allowance)



def _calculate_easter(year: int) -> date:
    """Anonymous Gregorian algorithm to compute Easter Sunday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _get_nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """Return the nth occurrence of a weekday in a month (weekday: 0=Monday..6=Sunday, n: 1-indexed)."""
    first_day = date(year, month, 1)
    first_weekday = first_day.weekday()
    day_offset = (weekday - first_weekday) % 7
    first_target = first_day + timedelta(days=day_offset)
    return first_target + timedelta(weeks=n - 1)


def _get_last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of a weekday in a month."""
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)
    day_offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=day_offset)


def _get_observed_holiday(target_date: date) -> date:
    """If holiday falls on Saturday -> observed Friday; if Sunday -> observed Monday."""
    if target_date.weekday() == 5:  # Saturday
        return target_date - timedelta(days=1)
    elif target_date.weekday() == 6:  # Sunday
        return target_date + timedelta(days=1)
    return target_date


def is_nyse_holiday(d: date) -> bool:
    """Check if a date is an official NYSE market holiday."""
    year = d.year
    holidays = {
        _get_observed_holiday(date(year, 1, 1)),                  # New Year's Day
        _get_nth_weekday_of_month(year, 1, 0, 3),                 # Martin Luther King Jr. Day (3rd Mon in Jan)
        _get_nth_weekday_of_month(year, 2, 0, 3),                 # Washington's Birthday (3rd Mon in Feb)
        _calculate_easter(year) - timedelta(days=2),              # Good Friday (2 days before Easter)
        _get_last_weekday_of_month(year, 5, 0),                   # Memorial Day (Last Mon in May)
        _get_observed_holiday(date(year, 6, 19)),                 # Juneteenth
        _get_observed_holiday(date(year, 7, 4)),                  # Independence Day
        _get_nth_weekday_of_month(year, 9, 0, 1),                 # Labor Day (1st Mon in Sep)
        _get_nth_weekday_of_month(year, 11, 3, 4),                # Thanksgiving (4th Thu in Nov)
        _get_observed_holiday(date(year, 12, 25)),                # Christmas Day
    }
    return d in holidays


def get_previous_active_trading_day(ref_date: date) -> date:
    """Find the most recent past trading day (Mon-Fri and not a holiday)."""
    cur = ref_date - timedelta(days=1)
    while cur.weekday() >= 5 or is_nyse_holiday(cur):
        cur -= timedelta(days=1)
    return cur


def compute_session_window(reference_time: datetime | None = None) -> SessionWindow:
    """
    Computes the exact daily-session window anchored to America/New_York.
    
    Session Rules:
    - Regular Trading Day (Mon-Fri, non-holiday):
      Window Start = 00:00:00 ET of current day (Midnight ET).
      Window End = Current run time.
    - Weekend / Market Holiday:
      Window Start = 16:00:00 ET of the prior active trading day,
                     capped at max 72 hours prior to run time.
      Window End = Current run time.
    """
    if reference_time is None:
        run_utc = datetime.now(timezone.utc)
    else:
        if reference_time.tzinfo is None:
            run_utc = reference_time.replace(tzinfo=timezone.utc)
        else:
            run_utc = reference_time.astimezone(timezone.utc)

    run_et = run_utc.astimezone(EASTERN_TZ)
    current_date_et = run_et.date()
    current_time_et = run_et.time()

    is_weekend = run_et.weekday() >= 5  # 5=Sat, 6=Sun
    is_holiday = is_nyse_holiday(current_date_et)

    if is_weekend or is_holiday:
        session_type = SessionType.WEEKEND_SESSION if is_weekend else SessionType.HOLIDAY_OFF_MARKET
        is_market_open = False

        # Anchor to 16:00 ET of the previous active trading day
        prev_trading_day = get_previous_active_trading_day(current_date_et)
        uncapped_start_et = datetime.combine(
            prev_trading_day, time(16, 0, 0), tzinfo=EASTERN_TZ
        )
        
        # Enforce maximum 72-hour lookback cap
        earliest_allowed_et = run_et - timedelta(hours=72)
        window_start_et = max(uncapped_start_et, earliest_allowed_et)
        window_end_et = run_et
    else:
        # Trading Day
        t_pre = time(4, 0, 0)
        t_open = time(9, 30, 0)
        t_close = time(16, 0, 0)
        t_post = time(20, 0, 0)

        if t_open <= current_time_et <= t_close:
            session_type = SessionType.REGULAR_HOURS
            is_market_open = True
        elif t_pre <= current_time_et < t_open:
            session_type = SessionType.PRE_MARKET
            is_market_open = False
        elif t_close < current_time_et <= t_post:
            session_type = SessionType.POST_MARKET
            is_market_open = False
        else:
            session_type = SessionType.OVERNIGHT_PRE_OPEN
            is_market_open = False

        # Daily window starts at midnight ET (00:00:00) of the current day
        window_start_et = datetime.combine(
            current_date_et, time(0, 0, 0), tzinfo=EASTERN_TZ
        )
        window_end_et = run_et

    window_start_utc = window_start_et.astimezone(timezone.utc)
    window_end_utc = window_end_et.astimezone(timezone.utc)
    duration_hours = round((window_end_utc - window_start_utc).total_seconds() / 3600.0, 2)

    return SessionWindow(
        market_timezone="America/New_York",
        session_type=session_type,
        is_market_open=is_market_open,
        window_start_utc=window_start_utc,
        window_start_et=window_start_et,
        window_end_utc=window_end_utc,
        window_end_et=window_end_et,
        window_duration_hours=duration_hours,
        run_timestamp_utc=run_utc,
        run_timestamp_et=run_et,
    )


def compute_session_bucket(session: SessionWindow, now: datetime | None = None) -> tuple[str, int]:
    """
    Computes deterministic session cache bucket string and bucket duration in seconds.
    - REGULAR_HOURS: floored to CACHE_INTRADAY_BUCKET_MINUTES
    - Non-regular sessions: floored to CACHE_OFF_MARKET_BUCKET_MINUTES
    Returns: (bucket_str, bucket_seconds)
    """
    from app.config import settings
    ref_dt = now or session.run_timestamp_utc
    if ref_dt.tzinfo is None:
        ref_dt = ref_dt.replace(tzinfo=timezone.utc)
    ref_et = ref_dt.astimezone(EASTERN_TZ)

    if session.session_type == SessionType.REGULAR_HOURS:
        bucket_mins = settings.CACHE_INTRADAY_BUCKET_MINUTES
    else:
        bucket_mins = settings.CACHE_OFF_MARKET_BUCKET_MINUTES

    floored_minute = (ref_et.minute // bucket_mins) * bucket_mins
    floored_et = ref_et.replace(minute=floored_minute, second=0, microsecond=0)
    bucket_str = f"{session.session_type.value}_{floored_et.strftime('%Y%m%d%H%M')}"
    bucket_seconds = bucket_mins * 60
    return bucket_str, bucket_seconds


