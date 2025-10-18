# NFL Data Source Information

## Using nflreadpy (Updated 2024)

This project uses **`nflreadpy`**, the modern successor to the deprecated `nfl-data-py` library.

### Quick Facts

- **Library**: `nflreadpy` (https://github.com/greerreNFL/nflreadpy)
- **Status**: ✅ Actively maintained (2024+)
- **Replaces**: `nfl-data-py` (deprecated)
- **License**: MIT
- **Data Coverage**: 1999-present

### Installation

```bash
pip install nflreadpy
```

### What Data Is Available?

#### 1. Weekly Player Statistics (Primary Data Source)
```python
import nflreadpy as nfl

# Load weekly stats for 2024 season
df = nfl.load_player_stats([2024])
```

**Includes:**
- Player identification (ID, name, position, team)
- Passing stats (completions, attempts, yards, TDs, INTs)
- Rushing stats (carries, yards, TDs)
- Receiving stats (targets, receptions, yards, TDs)
- Fumbles, 2-point conversions

#### 2. Play-by-Play Data (Advanced Features)
```python
pbp = nfl.load_pbp([2024])
```

**Includes:**
- Every play from every game
- EPA (Expected Points Added)
- Success rate
- Air yards, YAC
- Route details

#### 3. Game Schedules (Context Features)
```python
schedule = nfl.load_schedules([2024])
```

**Includes:**
- Game IDs, teams, dates
- Betting lines (spread, over/under)
- Stadium info (dome/outdoor, roof type)
- Weather (temperature, wind)

#### 4. Rosters (Player Info)
```python
rosters = nfl.load_rosters([2024])
```

**Includes:**
- Player demographics
- Draft information
- Physical attributes

### Data Freshness

| Time | Status |
|------|--------|
| **During season** | Updated weekly after games complete |
| **Off-season** | Historical data available immediately |
| **2024 Season** | Through most recent completed week (~Week 7) |

### Why nflreadpy?

✅ **Advantages:**
1. **Free**: No API keys or subscriptions
2. **Comprehensive**: All public NFL data
3. **Reliable**: Community-maintained, actively updated
4. **Well-documented**: Clear API, examples
5. **Fast**: Efficient data loading with caching
6. **Pythonic**: pandas DataFrames out of the box

### Migration from nfl-data-py

If you previously used `nfl-data-py`, the transition requires minor API changes:

```python
# Old (deprecated)
# import nfl_data_py as nfl
# df = nfl.import_weekly_data([2024])
# pbp = nfl.import_pbp_data([2024])
# rosters = nfl.import_seasonal_rosters([2024])
# schedules = nfl.import_schedules([2024])

# New (current) - slightly different function names
import nflreadpy as nfl
df = nfl.load_player_stats([2024])  # Changed from import_weekly_data
pbp = nfl.load_pbp([2024])          # Changed from import_pbp_data
rosters = nfl.load_rosters([2024])  # Changed from import_seasonal_rosters
schedules = nfl.load_schedules([2024])  # Same basic name
```

### Data Storage

The project caches downloaded data locally:

```
data/
├── raw/            # Original downloads (kept for reference)
├── interim/        # Cleaned/merged data from nflreadpy
└── processed/      # Model-ready features
```

**Cache files** (automatically created):
- `data/interim/weekly_stats_2019_2024.parquet` (~100 MB)
- `data/interim/schedules_2019_2024.parquet` (~5 MB)
- `data/interim/rosters_2019_2024.parquet` (~10 MB)

### 2024 Season Data

As of October 2024, the following weeks are available:

| Week | Games Completed | Data Available |
|------|----------------|----------------|
| 1-6 | ✅ Complete | ✅ Yes |
| 7 | ⏳ In progress | 🔄 Partial |
| 8+ | 🔮 Future | ❌ Not yet |

### Example Usage in This Project

```python
# src/ffproj/data.py
import nflreadpy as nfl
from .scoring import add_fantasy_points_columns

def load_weekly_stats(seasons):
    """Load and prepare NFL weekly data."""
    # Download from nflreadpy
    df = nfl.load_player_stats(seasons)

    # Filter to fantasy-relevant positions
    df = df[df['position'].isin(['QB', 'RB', 'WR', 'TE'])]

    # Calculate fantasy points
    df = add_fantasy_points_columns(df, scoring_systems=['ppr', 'half_ppr'])

    return df
```

### Common Issues & Solutions

#### Issue: Import error
```
ModuleNotFoundError: No module named 'nflreadpy'
```
**Solution:**
```bash
pip install nflreadpy
```

#### Issue: Slow first download
```
Downloading weekly stats for seasons [2019, 2020, 2021, 2022, 2023, 2024]...
```
**Solution:** This is normal! First download takes 5-10 minutes. Data is cached for future runs.

#### Issue: Missing 2024 data
**Solution:** Make sure you're requesting weeks that have been completed. Week 7+ may not be available yet.

### Alternative Data Sources

While `nflreadpy` is recommended, alternatives include:

| Source | Pros | Cons |
|--------|------|------|
| **Pro Football Reference** | Very detailed | Manual scraping required |
| **ESPN API** | Real-time | Unofficial, unstable |
| **FantasyData** | DFS-focused | Paid subscription |
| **NFL.com API** | Official | Limited public access |

For this project, **nflreadpy provides the best balance** of completeness, ease-of-use, and cost (free).

### Data License & Usage

- **nflreadpy**: MIT License
- **NFL Data**: Publicly available statistics
- **Usage**: Free for personal/educational use
- **Commercial**: Check NFL data usage policies

### References

- **nflreadpy GitHub**: https://github.com/greerreNFL/nflreadpy
- **nflreadpy Docs**: (see GitHub README)
- **Original nfl-data-py**: https://github.com/cooperdff/nfl_data_py (deprecated)
- **nflfastR (R version)**: https://www.nflfastr.com/

### Support

If you encounter issues with data loading:

1. Check your internet connection
2. Verify `nflreadpy` is installed: `pip list | grep nflreadpy`
3. Try clearing cache: `rm -rf data/interim/*.parquet`
4. Report bugs to: https://github.com/greerreNFL/nflreadpy/issues

---

**Last Updated**: October 2024
**nflreadpy Version**: 0.0.1+
**Maintained By**: NFL Analytics Community
