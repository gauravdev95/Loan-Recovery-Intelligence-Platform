import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Borrowers from "./pages/Borrowers";
import BorrowerDetail from "./pages/BorrowerDetail";
import RecoveryQueue from "./pages/RecoveryQueue";
import ModelPerformance from "./pages/ModelPerformance";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/borrowers" element={<Borrowers />} />
        <Route path="/borrowers/:id" element={<BorrowerDetail />} />
        <Route path="/recovery" element={<RecoveryQueue />} />
        <Route path="/performance" element={<ModelPerformance />} />
        <Route path="*" element={<Dashboard />} />
      </Route>
    </Routes>
  );
}
