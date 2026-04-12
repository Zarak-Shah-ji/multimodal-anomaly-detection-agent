"""
Builds and queries the Neo4j Knowledge Graph.

Schema:
  (:Sensor)-[:MEASURES]->(:Component)-[:HAS_FAILURE_MODE]->(:FailureMode)
  (:Sensor)-[:INDICATES]->(:FailureMode)
  (:Component)-[:PART_OF]->(:Subsystem)
"""

from neo4j import GraphDatabase

# ── Domain Knowledge ──────────────────────────────────────────────────────────

SENSORS = [
    {"id": "T2",      "name": "Fan Inlet Temperature",       "unit": "K",   "type": "temperature"},
    {"id": "T24",     "name": "LPC Outlet Temperature",      "unit": "K",   "type": "temperature"},
    {"id": "T30",     "name": "HPC Outlet Temperature",      "unit": "K",   "type": "temperature"},
    {"id": "T50",     "name": "LPT Outlet Temperature",      "unit": "K",   "type": "temperature"},
    {"id": "P2",      "name": "Fan Inlet Pressure",          "unit": "psi", "type": "pressure"},
    {"id": "P15",     "name": "Bypass Duct Pressure",        "unit": "psi", "type": "pressure"},
    {"id": "P30",     "name": "HPC Outlet Pressure",         "unit": "psi", "type": "pressure"},
    {"id": "Nf",      "name": "Physical Fan Speed",          "unit": "rpm", "type": "speed"},
    {"id": "Nc",      "name": "Physical Core Speed",         "unit": "rpm", "type": "speed"},
    {"id": "epr",     "name": "Engine Pressure Ratio",       "unit": "",    "type": "ratio"},
    {"id": "Ps30",    "name": "HPC Static Pressure",         "unit": "psi", "type": "pressure"},
    {"id": "phi",     "name": "Fuel Flow to Ps30 Ratio",     "unit": "",    "type": "ratio"},
    {"id": "NRf",     "name": "Corrected Fan Speed",         "unit": "rpm", "type": "speed"},
    {"id": "NRc",     "name": "Corrected Core Speed",        "unit": "rpm", "type": "speed"},
    {"id": "BPR",     "name": "Bypass Ratio",                "unit": "",    "type": "ratio"},
    {"id": "farB",    "name": "Burner Fuel-Air Ratio",       "unit": "",    "type": "ratio"},
    {"id": "htBleed", "name": "Bleed Enthalpy",              "unit": "",    "type": "enthalpy"},
    {"id": "Nf_dmd",  "name": "Demanded Fan Speed",          "unit": "rpm", "type": "speed"},
    {"id": "PCNfR",   "name": "Corrected Fan Speed at Inlet","unit": "%",   "type": "speed"},
    {"id": "W31",     "name": "HPT Coolant Bleed",           "unit": "pps", "type": "flow"},
    {"id": "W32",     "name": "LPT Coolant Bleed",           "unit": "pps", "type": "flow"},
]

COMPONENTS = [
    {"id": "fan",        "name": "Fan",                         "subsystem": "cold_section"},
    {"id": "lpc",        "name": "Low Pressure Compressor",     "subsystem": "cold_section"},
    {"id": "hpc",        "name": "High Pressure Compressor",    "subsystem": "hot_section"},
    {"id": "combustor",  "name": "Combustor",                   "subsystem": "hot_section"},
    {"id": "hpt",        "name": "High Pressure Turbine",       "subsystem": "hot_section"},
    {"id": "lpt",        "name": "Low Pressure Turbine",        "subsystem": "hot_section"},
    {"id": "bypass_duct","name": "Bypass Duct",                 "subsystem": "cold_section"},
]

FAILURE_MODES = [
    {"id": "tip_wear",      "name": "Tip Clearance Wear",    "severity": 3,
     "description": "Progressive wear of blade tips leading to increased tip clearance and reduced efficiency."},
    {"id": "fouling",       "name": "Compressor Fouling",    "severity": 2,
     "description": "Deposit buildup on blade surfaces reducing aerodynamic performance."},
    {"id": "erosion",       "name": "Blade Erosion",         "severity": 4,
     "description": "Material loss from blade surfaces due to particle impact."},
    {"id": "thermal_degr",  "name": "Thermal Degradation",   "severity": 4,
     "description": "High-temperature damage to turbine coatings and base material."},
    {"id": "seal_leakage",  "name": "Seal Leakage",          "severity": 3,
     "description": "Leakage past seals causing reduced pressure ratios and efficiency."},
    {"id": "fuel_nozzle",   "name": "Fuel Nozzle Blockage",  "severity": 3,
     "description": "Partial blockage of fuel nozzles causing uneven combustion."},
    {"id": "bearing_wear",  "name": "Bearing Wear",          "severity": 5,
     "description": "Wear of shaft bearings leading to vibration and eventual failure."},
    {"id": "cooling_loss",  "name": "Cooling Flow Reduction","severity": 4,
     "description": "Reduced cooling airflow to turbine blades risking thermal damage."},
]

# Sensor → Component mappings (which sensors measure which component)
SENSOR_COMPONENT = {
    "T2":      ["fan"],
    "T24":     ["lpc"],
    "T30":     ["hpc"],
    "T50":     ["lpt"],
    "P2":      ["fan"],
    "P15":     ["bypass_duct"],
    "P30":     ["hpc"],
    "Nf":      ["fan", "lpc", "lpt"],
    "Nc":      ["hpc", "hpt"],
    "epr":     ["fan", "hpt"],
    "Ps30":    ["hpc"],
    "phi":     ["combustor"],
    "NRf":     ["fan"],
    "NRc":     ["hpc"],
    "BPR":     ["bypass_duct", "fan"],
    "farB":    ["combustor"],
    "htBleed": ["hpc", "hpt"],
    "Nf_dmd":  ["fan"],
    "PCNfR":   ["fan"],
    "W31":     ["hpt"],
    "W32":     ["lpt"],
}

# Component → Failure Mode mappings
COMPONENT_FAILURE = {
    "fan":        ["tip_wear", "fouling", "erosion", "bearing_wear"],
    "lpc":        ["tip_wear", "fouling", "erosion"],
    "hpc":        ["tip_wear", "fouling", "erosion", "seal_leakage"],
    "combustor":  ["fuel_nozzle", "thermal_degr"],
    "hpt":        ["thermal_degr", "tip_wear", "cooling_loss", "erosion"],
    "lpt":        ["thermal_degr", "tip_wear", "cooling_loss"],
    "bypass_duct":["seal_leakage", "fouling"],
}

# Sensor → directly indicates failure mode
SENSOR_FAILURE = {
    "T50":     ["thermal_degr", "cooling_loss"],
    "T30":     ["fouling", "seal_leakage"],
    "P30":     ["fouling", "tip_wear"],
    "Ps30":    ["seal_leakage", "tip_wear"],
    "W31":     ["cooling_loss", "thermal_degr"],
    "W32":     ["cooling_loss", "thermal_degr"],
    "phi":     ["fuel_nozzle"],
    "NRf":     ["bearing_wear", "tip_wear"],
    "BPR":     ["seal_leakage", "tip_wear"],
}


# ── KG Builder ────────────────────────────────────────────────────────────────

class KGBuilder:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def build(self):
        with self.driver.session() as session:
            session.execute_write(self._clear_db)
            session.execute_write(self._create_sensors)
            session.execute_write(self._create_components)
            session.execute_write(self._create_failure_modes)
            session.execute_write(self._create_sensor_component_rels)
            session.execute_write(self._create_component_failure_rels)
            session.execute_write(self._create_sensor_failure_rels)
        print("KG built successfully.")

    @staticmethod
    def _clear_db(tx):
        tx.run("MATCH (n) DETACH DELETE n")

    @staticmethod
    def _create_sensors(tx):
        for s in SENSORS:
            tx.run(
                "CREATE (:Sensor {id: $id, name: $name, unit: $unit, type: $type})",
                **s
            )

    @staticmethod
    def _create_components(tx):
        for c in COMPONENTS:
            tx.run(
                "CREATE (:Component {id: $id, name: $name, subsystem: $subsystem})",
                **c
            )

    @staticmethod
    def _create_failure_modes(tx):
        for fm in FAILURE_MODES:
            tx.run(
                "CREATE (:FailureMode {id: $id, name: $name, severity: $severity, description: $description})",
                **fm
            )

    @staticmethod
    def _create_sensor_component_rels(tx):
        for sensor_id, comp_ids in SENSOR_COMPONENT.items():
            for comp_id in comp_ids:
                tx.run(
                    """
                    MATCH (s:Sensor {id: $sid}), (c:Component {id: $cid})
                    CREATE (s)-[:MEASURES]->(c)
                    """,
                    sid=sensor_id, cid=comp_id
                )

    @staticmethod
    def _create_component_failure_rels(tx):
        for comp_id, fm_ids in COMPONENT_FAILURE.items():
            for fm_id in fm_ids:
                tx.run(
                    """
                    MATCH (c:Component {id: $cid}), (fm:FailureMode {id: $fmid})
                    CREATE (c)-[:HAS_FAILURE_MODE]->(fm)
                    """,
                    cid=comp_id, fmid=fm_id
                )

    @staticmethod
    def _create_sensor_failure_rels(tx):
        for sensor_id, fm_ids in SENSOR_FAILURE.items():
            for fm_id in fm_ids:
                tx.run(
                    """
                    MATCH (s:Sensor {id: $sid}), (fm:FailureMode {id: $fmid})
                    CREATE (s)-[:INDICATES]->(fm)
                    """,
                    sid=sensor_id, fmid=fm_id
                )


# ── KG Query Interface ────────────────────────────────────────────────────────

def query_kg_for_sensors(driver, sensor_ids: list[str]) -> list[dict]:
    """
    Given a list of anomalous sensor IDs, return related components + failure modes.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (s:Sensor)-[:MEASURES]->(c:Component)-[:HAS_FAILURE_MODE]->(fm:FailureMode)
            WHERE s.id IN $sensor_ids
            RETURN s.id AS sensor, s.name AS sensor_name,
                   c.id AS component, c.name AS component_name,
                   fm.name AS failure_mode, fm.severity AS severity,
                   fm.description AS fm_description
            ORDER BY fm.severity DESC
            """,
            sensor_ids=sensor_ids
        )
        return [dict(r) for r in result]


def query_direct_failure_indicators(driver, sensor_ids: list[str]) -> list[dict]:
    """
    Return failure modes directly indicated by the anomalous sensors.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (s:Sensor)-[:INDICATES]->(fm:FailureMode)
            WHERE s.id IN $sensor_ids
            RETURN s.id AS sensor, s.name AS sensor_name,
                   fm.name AS failure_mode, fm.severity AS severity,
                   fm.description AS fm_description
            ORDER BY fm.severity DESC
            """,
            sensor_ids=sensor_ids
        )
        return [dict(r) for r in result]


def format_kg_context(component_results: list[dict], direct_results: list[dict]) -> str:
    """Format KG query results as readable text for the report agent."""
    lines = ["=== Knowledge Graph Context ===\n"]

    if direct_results:
        lines.append("## Direct Failure Indicators (from anomalous sensors):")
        seen = set()
        for r in direct_results:
            key = (r["sensor"], r["failure_mode"])
            if key not in seen:
                seen.add(key)
                lines.append(
                    f"  • Sensor [{r['sensor']} - {r['sensor_name']}] "
                    f"→ INDICATES → [{r['failure_mode']}] (severity {r['severity']}/5)\n"
                    f"    {r['fm_description']}"
                )
        lines.append("")

    if component_results:
        lines.append("## Component-Level Failure Modes:")
        seen_comp = set()
        for r in component_results:
            key = (r["component"], r["failure_mode"])
            if key not in seen_comp:
                seen_comp.add(key)
                lines.append(
                    f"  • Component [{r['component_name']}] "
                    f"→ HAS_FAILURE_MODE → [{r['failure_mode']}] (severity {r['severity']}/5)\n"
                    f"    {r['fm_description']}"
                )

    return "\n".join(lines)


# ── Fallback (no Neo4j) ───────────────────────────────────────────────────────

def get_static_kg_context(sensor_ids: list[str]) -> str:
    """Static fallback when Neo4j is unavailable."""
    lines = ["=== Knowledge Graph Context (static fallback) ===\n"]
    for sid in sensor_ids:
        comp_ids = SENSOR_COMPONENT.get(sid, [])
        fm_ids   = SENSOR_FAILURE.get(sid, [])
        comp_names = [c["name"] for c in COMPONENTS if c["id"] in comp_ids]
        fm_data    = [f for f in FAILURE_MODES if f["id"] in fm_ids]
        if comp_names:
            lines.append(f"Sensor {sid} measures: {', '.join(comp_names)}")
        for fm in fm_data:
            lines.append(
                f"  → Indicates: {fm['name']} (severity {fm['severity']}/5) — {fm['description']}"
            )
    return "\n".join(lines)
