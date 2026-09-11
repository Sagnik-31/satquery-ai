import { createRoot } from "react-dom/client";
import Home from "./page";

import "leaflet/dist/leaflet.css";
import "./globals.css";

createRoot(document.getElementById("root")).render(<Home />);
