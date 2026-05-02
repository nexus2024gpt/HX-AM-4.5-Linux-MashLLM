#!/usr/bin/env python3
# mgap_lib/scripts/init_db.py — MGAP Library v1.0
"""
Инициализация базы данных MGAP из mgap_registry.json + unesco_taxonomy.json.

Что делает:
  1. Создаёт таблицы (если не существуют)
  2. Заполняет Discipline, Sector из UNESCO-таксономии
  3. Загружает MGAPModel из mgap_registry.json
  4. Создаёт Specialization для каждого найденного типа

CLI:
  python mgap_lib/scripts/init_db.py
  python mgap_lib/scripts/init_db.py --db sqlite:///mgap.db
  python mgap_lib/scripts/init_db.py --registry /path/to/mgap_registry.json
  python mgap_lib/scripts/init_db.py --reset   # удалить и пересоздать
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Чтобы скрипт работал из любого места
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MGAP.init_db")

UNESCO_TAX_PATH  = Path(__file__).parent.parent / "data" / "unesco_taxonomy.json"
REGISTRY_DEFAULT = Path("mgap_registry.json")


def load_taxonomy(path: Path) -> dict:
    if not path.exists():
        logger.warning(f"unesco_taxonomy.json not found at {path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def populate_taxonomy(session, taxonomy: dict):
    """Загружает дисциплины и сектора из UNESCO-таксономии."""
    from mgap_lib.models.database import Discipline, Sector, Specialization

    disciplines = taxonomy.get("disciplines", [])
    disc_count = sec_count = spec_count = 0

    for disc_data in disciplines:
        code     = disc_data.get("code")
        name_ru  = disc_data.get("name_ru", "")
        name_en  = disc_data.get("name_en", "")
        if not code:
            continue

        existing = session.query(Discipline).filter_by(code=code).first()
        if existing is None:
            disc = Discipline(code=code, name_ru=name_ru, name_en=name_en)
            session.add(disc)
            disc_count += 1
        else:
            existing.name_ru = name_ru
            existing.name_en = name_en

        for sector_data in disc_data.get("sectors", []):
            s_code   = sector_data.get("code")
            s_name_ru = sector_data.get("name_ru", "")
            s_name_en = sector_data.get("name_en", "")
            if not s_code:
                continue

            existing_s = session.query(Sector).filter_by(code=s_code).first()
            if existing_s is None:
                sec = Sector(code=s_code, name_ru=s_name_ru, name_en=s_name_en, disc_code=code)
                session.add(sec)
                sec_count += 1

            # Специализации из таксономии
            for spec_name in sector_data.get("specializations", []):
                existing_sp = session.query(Specialization).filter_by(
                    name=spec_name, sector_code=s_code
                ).first()
                if existing_sp is None:
                    sp = Specialization(name=spec_name, sector_code=s_code)
                    session.add(sp)
                    spec_count += 1

    session.commit()
    logger.info(f"Taxonomy: {disc_count} disciplines, {sec_count} sectors, {spec_count} specializations added")


def populate_models(session, registry_path: Path):
    """Загружает MGAPModel из JSON-реестра."""
    from mgap_lib.engine.registry import RegistryLoader
    loader = RegistryLoader(registry_path=registry_path)
    count  = loader.upsert_from_json(session, registry_path)
    return count


def main():
    parser = argparse.ArgumentParser(description="HX-AM MGAP Library — DB Init")
    parser.add_argument("--db",       type=str, default="sqlite:///mgap.db",
                        help="Database URL (default: sqlite:///mgap.db)")
    parser.add_argument("--registry", type=str, default=str(REGISTRY_DEFAULT),
                        help="Path to mgap_registry.json")
    parser.add_argument("--taxonomy", type=str, default=str(UNESCO_TAX_PATH),
                        help="Path to unesco_taxonomy.json")
    parser.add_argument("--reset",    action="store_true",
                        help="Drop and recreate all tables")
    parser.add_argument("--models-only", action="store_true",
                        help="Skip taxonomy, only load models")
    args = parser.parse_args()

    logger.info(f"MGAP DB Init — db_url={args.db}")

    from mgap_lib.models.database import (
        create_db_engine, create_tables, get_session_factory, Base
    )

    engine = create_db_engine(args.db)

    if args.reset:
        logger.warning("--reset: dropping all tables...")
        Base.metadata.drop_all(engine)
        logger.info("Tables dropped.")

    create_tables(engine)
    logger.info("Tables created / verified.")

    factory = get_session_factory(engine)
    session = factory()

    try:
        if not args.models_only:
            taxonomy = load_taxonomy(Path(args.taxonomy))
            if taxonomy:
                populate_taxonomy(session, taxonomy)
            else:
                logger.warning("Taxonomy empty or not found — skipping")

        registry_path = Path(args.registry)
        if registry_path.exists():
            count = populate_models(session, registry_path)
            logger.info(f"Registry: {count} models synced to DB")
        else:
            logger.error(f"Registry file not found: {registry_path}")
            sys.exit(1)

    except Exception as e:
        session.rollback()
        logger.error(f"Init failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        session.close()

    logger.info("✅ MGAP DB initialization complete!")
    print(f"\n✅ База данных готова: {args.db}")
    print(f"   Следующий шаг:")
    print(f"   python mgap_lib/cli/mgap_cli.py match --artifact <artifact_id>")


if __name__ == "__main__":
    main()
