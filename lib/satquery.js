export const scenes = [
    {
        id: "bengaluru",
        name: "Bengaluru",
        region: "Karnataka, India",
        center: [12.955, 77.67],
        zoom: 12,
        bounds: [
            [12.9, 77.6],
            [13.01, 77.74],
        ],
    },
    {
        id: "kochi",
        name: "Kochi",
        region: "Kerala, India",
        center: [9.975, 76.29],
        zoom: 12,
        bounds: [
            [9.92, 76.23],
            [10.03, 76.36],
        ],
    },
    {
        id: "hyderabad",
        name: "Hyderabad",
        region: "Telangana, India",
        center: [17.43, 78.45],
        zoom: 12,
        bounds: [
            [17.37, 78.38],
            [17.49, 78.52],
        ],
    },
    {
        id: "delhi",
        name: "Delhi",
        region: "Delhi, India",
        center: [28.6139, 77.2090],
        zoom: 12,
        bounds: [
            [28.55, 77.14],
            [28.67, 77.26],
        ],
    },
    {
        id: "mumbai",
        name: "Mumbai",
        region: "Maharashtra, India",
        center: [19.0760, 72.8777],
        zoom: 12,
        bounds: [
            [19.01, 72.80],
            [19.14, 72.95],
        ],
    },
    {
        id: "kolkata",
        name: "Kolkata",
        region: "West Bengal, India",
        center: [22.5726, 88.3639],
        zoom: 12,
        bounds: [
            [22.50, 88.28],
            [22.65, 88.45],
        ],
    },
];
const polygon = (
    name,
    kind,
    coords,
) => ({
    type: "Feature",
    properties: {
        name,
        kind,
        source: "Illustrative geometry",
        confidence: null,
    },
    geometry: {type: "Polygon", coordinates: [[...coords, coords[0]]]},
});

export function demoFeatures(scene, kind) {
    const [lat, lng] = scene.center;
    const water =
        scene.id === "bengaluru"
            ? [
                [77.649, 12.937],
                [77.653, 12.946],
                [77.664, 12.951],
                [77.68, 12.946],
                [77.674, 12.936],
                [77.66, 12.93],
            ]
            : [
                [lng - 0.021, lat - 0.015],
                [lng - 0.015, lat + 0.013],
                [lng - 0.006, lat + 0.021],
                [lng + 0.003, lat + 0.007],
                [lng - 0.006, lat - 0.022],
            ];
    const items = [
        polygon(
            scene.id === "bengaluru"
                ? "Bellandur lake"
                : "Water region",
            "water",
            water,
        ),
        polygon("Built-up cluster A", "urban", [
            [lng + 0.006, lat + 0.012],
            [lng + 0.023, lat + 0.013],
            [lng + 0.025, lat + 0.025],
            [lng + 0.011, lat + 0.029],
        ]),
        polygon("Built-up cluster B", "urban", [
            [lng + 0.021, lat - 0.009],
            [lng + 0.038, lat - 0.007],
            [lng + 0.037, lat + 0.005],
            [lng + 0.024, lat + 0.005],
        ]),
        polygon("Vegetation region", "vegetation", [
            [lng - 0.036, lat + 0.013],
            [lng - 0.028, lat + 0.016],
            [lng - 0.021, lat + 0.032],
            [lng - 0.038, lat + 0.034],
        ]),
    ];
    return {
        type: "FeatureCollection",
        features: items.filter((f) => !kind || f.properties?.kind === kind),
    };
}

export function getDemoResult(query, mode, scene) {
    const q = query.toLowerCase();
    const kind = /water|flood|lake/.test(q)
        ? "water"
        : /build|urban|road/.test(q)
            ? "urban"
            : /forest|vegetation|crop|green/.test(q)
                ? "vegetation"
                : undefined;
    const change = /chang|increas|decreas|before|after/.test(q);
    const fusion = /sar|fusion|both|radar/.test(q);
    if (change && mode !== "temporal")
        throw new Error(
            "Choose Before / after to run a change query with two dates.",
        );
    if (fusion && mode !== "fusion")
        throw new Error("Choose Optical + SAR to run a paired-sensor query.");
    const task =
        mode === "temporal"
            ? "Change understanding"
            : mode === "fusion"
                ? "Cross-modal analysis"
                : /highlight|locate|show|outline/.test(q)
                    ? "Region grounding"
                    : /describe|caption/.test(q)
                        ? "Scene description"
                        : "Visual question answering";
    const answer =
        mode === "temporal"
            ? "Two built-up regions are highlighted on the map. Select an outline to inspect the region."
            : mode === "fusion"
                ? "The optical and SAR example includes the highlighted regions. Select an outline for details."
                : kind === "water"
                    ? "A water region is highlighted. Select the outline for details."
                    : kind === "vegetation"
                        ? "A vegetation region is highlighted in green. Select the outline for details."
                        : "The overlays show water, built-up, and vegetation regions. Select an outline for details.";
    const features = demoFeatures(scene, mode === "temporal" ? "urban" : kind);
    return {
        title:
            mode === "temporal"
                ? "Change regions"
                : mode === "fusion"
                    ? "Optical and SAR overview"
                    : kind === "water"
                        ? "Water regions"
                        : "Scene overview",
        answer,
        task,
        features,
        simulated: true,
        metrics: [
            {label: "Regions", value: String(features.features.length)},
            {label: "Format", value: "GeoJSON"},
        ],
        trace: [
            `Input configuration: ${mode}`,
            `Task selected: ${task}`,
            `Output: ${features.features.length} GeoJSON regions`,
        ],
    };
}

export function validateGeoJSON(value) {
    if (!value || typeof value !== "object")
        throw new Error("Expected a GeoJSON FeatureCollection.");
    const fc = value;
    if (
        fc.type !== "FeatureCollection" ||
        !Array.isArray(fc.features) ||
        fc.features.length > 10000
    )
        throw new Error("Use a FeatureCollection with at most 10,000 features.");
    const position = (p) =>
        Array.isArray(p) &&
        p.length >= 2 &&
        p.slice(0, 2).every((n) => typeof n === "number" && Number.isFinite(n)) &&
        Math.abs(p[0]) <= 180 &&
        Math.abs(p[1]) <= 90;
    const nested = (p, depth) =>
        depth === 0
            ? position(p)
            : Array.isArray(p) &&
            p.length > 0 &&
            p.every((x) => nested(x, depth - 1));
    const geometry = (g) => {
        if (!g) return false;
        if (g.type === "GeometryCollection") return g.geometries.every(geometry);
        const depths = {
            Point: 0,
            MultiPoint: 1,
            LineString: 1,
            MultiLineString: 2,
            Polygon: 2,
            MultiPolygon: 3,
        };
        if (!(g.type in depths) || !nested(g.coordinates, depths[g.type]))
            return false;
        if (g.type === "Polygon" || g.type === "MultiPolygon") {
            const polys = g.type === "Polygon" ? [g.coordinates] : g.coordinates;
            return polys.every((poly) =>
                poly.every(
                    (r) =>
                        r.length >= 4 &&
                        r[0][0] === r[r.length - 1][0] &&
                        r[0][1] === r[r.length - 1][1],
                ),
            );
        }
        return true;
    };
    if (!fc.features.every((f) => f.type === "Feature" && geometry(f.geometry)))
        throw new Error(
            "Invalid geometry.",
        );
    return fc;
}

export function download(
    name,
    data,
    type = "application/json",
) {
    const url = URL.createObjectURL(new Blob([data], {type}));
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function boundsToStacBbox(bounds) {
    if (!bounds || bounds.length !== 2) return null;
    const [[south, west], [north, east]] = bounds;
    return [west, south, east, north];
}

export function validatePair(images, mode) {
    const count = mode === "single" ? 1 : 2;
    if (images.length !== count)
        throw new Error(
            `Add ${count} image${count === 1 ? "" : "s"} for this workflow.`,
        );
    if (mode === "single") return;
    const [a, b] = images;
    if (mode === "fusion" && a.modality === b.modality)
        throw new Error("Choose one optical image and one SAR image.");

    if (mode === "temporal") {
        // The AI backend can compare ordinary RGB uploads when exact GIS
        // metadata is unavailable. Prefer georeferenced/co-registered pairs,
        // but do not block the hackathon demo for unreferenced PNG/JPEG files.
        if (a.modality !== b.modality)
            throw new Error("Use the same sensor modality for before/after analysis.");
        if (a.date && b.date && a.date === b.date)
            throw new Error("Use two different acquisition dates, or leave the dates blank for image-only comparison.");
    } else if (mode === "fusion") {
        if (!a.bounds || !b.bounds)
            throw new Error("Optical + SAR requires georeferenced images.");
    }
}
