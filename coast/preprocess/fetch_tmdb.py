import argparse
import json
import os
import time

import pandas as pd

from coast.config import get_dataset

META_COLS = ["parent_asin", "title", "features", "description"]

def needed_movie_ids(cfg):
    train = pd.read_csv(cfg.train_csv(), usecols=["parent_asin"])
    test = pd.read_csv(cfg.test_csv(), usecols=["parent_asin"])
    return set(train["parent_asin"].astype(str)).union(test["parent_asin"].astype(str))

def load_links(cfg):
    links = pd.read_csv(cfg.links_csv())
    links["movieId"] = links["movieId"].astype(str)
    links["tmdbId"] = links["tmdbId"].fillna(0).astype(int)
    return links.set_index("movieId")["tmdbId"].to_dict()

def load_movies_dat(cfg):
    rows = []
    with open(cfg.movies_dat(), encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::", 2)
            if len(parts) != 3:
                continue
            mid, title, genres = parts
            rows.append(
                {
                    "parent_asin": str(mid),
                    "title": title,
                    "features": genres.replace("|", " "),
                    "description": "",
                }
            )
    return pd.DataFrame(rows)

def load_cache(path):
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    return {}

def save_cache(path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f)

def fetch_tmdb_overview(tmdb_id, api_key, session):
    import requests

    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    r = session.get(url, params={"api_key": api_key}, timeout=30)
    if r.status_code == 404:
        return ""
    r.raise_for_status()
    data = r.json()
    overview = data.get("overview") or ""
    title = data.get("title") or ""
    genres = " ".join(g["name"] for g in data.get("genres", []))
    return overview, title, genres

def build_movielens_meta(cfg, out_path=None, sleep_s=0.26):
    import requests

    api_key = os.environ.get("TMDB_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "Set TMDB_API_KEY (free at https://www.themoviedb.org/settings/api). "
            "Without it, only movies.dat titles/genres are used."
        )

    out_path = out_path or cfg.meta_csv()
    needed = needed_movie_ids(cfg)
    movies = load_movies_dat(cfg)
    movies = movies[movies["parent_asin"].isin(needed)].copy()
    tmdb_map = load_links(cfg)
    cache = load_cache(cfg.tmdb_cache_path())
    session = requests.Session()

    for i, row in movies.iterrows():
        mid = row["parent_asin"]
        if mid in cache:
            continue
        tmdb_id = int(tmdb_map.get(mid, 0))
        if tmdb_id <= 0:
            cache[mid] = {"overview": "", "title": row["title"], "genres": row["features"]}
            continue
        try:
            overview, title, genres = fetch_tmdb_overview(tmdb_id, api_key, session)
            cache[mid] = {
                "overview": overview,
                "title": title or row["title"],
                "genres": genres or row["features"],
            }
        except Exception as e:
            print(f"warn tmdb {mid}/{tmdb_id}: {e}")
            cache[mid] = {"overview": "", "title": row["title"], "genres": row["features"]}
        if (i + 1) % 50 == 0:
            save_cache(cfg.tmdb_cache_path(), cache)
            print(f"  fetched {len(cache)} / {len(movies)} ...")
        time.sleep(sleep_s)

    save_cache(cfg.tmdb_cache_path(), cache)

    for mid, info in cache.items():
        mask = movies["parent_asin"] == mid
        if not mask.any():
            continue
        movies.loc[mask, "description"] = info.get("overview", "")
        if info.get("title"):
            movies.loc[mask, "title"] = info["title"]
        if info.get("genres"):
            movies.loc[mask, "features"] = info["genres"]

    out = movies[META_COLS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("saved", out_path, "shape", out.shape)
    with_overview = (out["description"].str.len() > 0).sum()
    print(f"items with plot text: {with_overview} / {len(out)}")

def build_meta_without_tmdb(cfg, out_path=None):
    needed = needed_movie_ids(cfg)
    movies = load_movies_dat(cfg)
    movies = movies[movies["parent_asin"].isin(needed)][META_COLS]
    out_path = out_path or cfg.meta_csv()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    movies.to_csv(out_path, index=False)
    print("saved (movies.dat only, no TMDB)", out_path, "shape", movies.shape)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="movielens", choices=["movielens"])
    p.add_argument(
        "--movies_only",
        action="store_true",
        help="skip TMDB API; use titles/genres from movies.dat only",
    )
    args = p.parse_args()
    cfg = get_dataset(args.dataset)

    if not cfg.train_csv().is_file():
        raise FileNotFoundError(
            f"run: python -m coast.preprocess.filter_splits --dataset {cfg.name} first"
        )
    if not cfg.movies_dat().is_file():
        raise FileNotFoundError("run: python -m coast.preprocess.download_movielens first")

    if args.movies_only or not os.environ.get("TMDB_API_KEY"):
        if not args.movies_only:
            print("TMDB_API_KEY not set — using movies.dat only")
        build_meta_without_tmdb(cfg)
    else:
        build_movielens_meta(cfg)

if __name__ == "__main__":
    main()
