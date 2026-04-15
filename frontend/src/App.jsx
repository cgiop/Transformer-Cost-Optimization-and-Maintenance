import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import { Upload, Cpu, Play, Settings, RefreshCw, Trash2, Calendar, User, Activity, Lock, Unlock, FolderUp, BarChart3 } from 'lucide-react';

const CircularProgress = ({ value, maxValue, label, subLabel, color, icon: Icon }) => {
  const radius = 50;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - ((value || 0) / maxValue) * circumference;

  return (
    <div className="card">
      <div className="card-title">{label}</div>
      <div className="card-value">{typeof value === 'number' ? value.toLocaleString() : value}</div>
      <div className="dials-container">
        <svg className="dial-circle" viewBox="0 0 120 120">
          <circle
            cx="60" cy="60" r={radius}
            stroke="var(--border-color)" strokeWidth="8" fill="none"
          />
          <circle
            cx="60" cy="60" r={radius}
            stroke={color} strokeWidth="8" fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            style={{ transition: 'stroke-dashoffset 0.5s ease-in-out', transform: 'rotate(-90deg)', transformOrigin: '50% 50%' }}
          />
          {Icon && (
            <foreignObject x="45" y="45" width="30" height="30">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%', color: color }}>
                <Icon size={18} />
              </div>
            </foreignObject>
          )}
        </svg>
      </div>
      <div className="card-subtitle">{subLabel}</div>
    </div>
  );
};

function App() {
  const [isAdmin, setIsAdmin] = useState(false);
  const [showGraph, setShowGraph] = useState(false);
  const [file, setFile] = useState(null);
  const [toast, setToast] = useState({ show: false, message: '', type: '' });
  const [isTraining, setIsTraining] = useState(false);
  const [predictionData, setPredictionData] = useState({
    Uptime_Percentage: 0.95,
    MTBF_Hours: 1200,
    MTTR_Hours: 8,
    Utilization_Rate: 0.85,
    Scheduled_Hours: 240,
    Downtime_Duration: 12,
    Maintenance_Parts_Cost: 1500,
    Energy_Consumption_kWh: 5000,
    Output_Quantity: 20000,
    Reject_Quantity: 150,
    Number_of_Breakdowns: 2
  });
  
  const [predictionResult, setPredictionResult] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [history, setHistory] = useState([
    { id: 1, action: "System Initialized", date: "Just now", status: "Ready" }
  ]);

  const showToast = (message, type = 'success') => {
    setToast({ show: true, message, type });
    setTimeout(() => setToast({ show: false, message: '', type: '' }), 3000);
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    try {
      await axios.post('/upload_data', formData);
      showToast('Dataset uploaded successfully');
      setHistory(prev => [{ id: Date.now(), action: "Upload Dataset", date: "Just now", status: "Success" }, ...prev]);
    } catch (err) {
      showToast('Upload failed', 'error');
    }
  };

  const handleTrain = async () => {
    setIsTraining(true);
    try {
      await axios.post('/train');
      showToast('Model trained successfully');
      setHistory(prev => [{ id: Date.now(), action: "Model Training", date: "Just now", status: "Success" }, ...prev]);
    } catch (err) {
      showToast('Training failed', 'error');
    } finally {
      setIsTraining(false);
    }
  };

  const handlePredict = async (e) => {
    if (e) e.preventDefault();
    try {
      const res = await axios.post('/predict_raw', predictionData);
      setPredictionResult(res.data);
      if (e) showToast('Prediction complete');
      
      const newChartPoint = {
        name: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}),
        cost: res.data.estimated_cost,
        profit: res.data.expected_profit
      };
      
      setChartData(prev => [...prev.slice(-6), newChartPoint]);
      setHistory(prev => [
        { id: Date.now(), action: `Predicted: ${res.data.recommended_action}`, date: "Just now", status: "Success" }, 
        ...prev
      ]);
    } catch (err) {
      if (e) showToast(err.response?.data?.detail || 'Prediction failed', 'error');
    }
  };

  useEffect(() => {
    handlePredict();
  }, []);

  const handleInputChange = (e) => {
    setPredictionData(prev => ({
      ...prev,
      [e.target.name]: parseFloat(e.target.value) || 0
    }));
  };

  const inputFields = [
    { name: 'Uptime_Percentage', label: 'Uptime Percentage', step: '0.01' },
    { name: 'MTBF_Hours', label: 'MTBF (Hours)', step: '1' },
    { name: 'MTTR_Hours', label: 'MTTR (Hours)', step: '1' },
    { name: 'Utilization_Rate', label: 'Utilization Rate', step: '0.01' },
    { name: 'Scheduled_Hours', label: 'Scheduled Hours', step: '1' },
    { name: 'Downtime_Duration', label: 'Downtime Duration', step: '1' },
    { name: 'Maintenance_Parts_Cost', label: 'Maint. Parts Cost ($)', step: '1' },
    { name: 'Energy_Consumption_kWh', label: 'Energy (kWh)', step: '1' },
    { name: 'Output_Quantity', label: 'Output Quantity', step: '1' },
    { name: 'Reject_Quantity', label: 'Reject Quantity', step: '1' },
    { name: 'Number_of_Breakdowns', label: 'Breakdowns', step: '1' }
  ];

  const qvals = predictionResult?.q_values;
  let confidences = [];
  if (qvals) {
    const sumAbs = qvals.reduce((a, b) => a + Math.abs(b), 0);
    confidences = qvals.map((v) => sumAbs !== 0 ? Math.abs(v) / sumAbs : 1/3);
  }
  const actionNames = ["Do Nothing", "Preventive Maintenance", "Corrective Maintenance"];
  const progressColors = ["var(--text-secondary)", "var(--primary-blue)", "var(--primary-magenta)"];

  return (
    <div className="app-container">
      <main className="main-content">
        <header className="dashboard-header">
          <h1>Transformer Cost Optimization and Maintenance</h1>
          <div className="flex-row">
            <button 
              className={`btn ${isAdmin ? 'btn-danger' : 'btn-filled'}`} 
              onClick={() => setIsAdmin(!isAdmin)}
            >
              {isAdmin ? <><Unlock size={14}/> Exit Admin Mode</> : <><Lock size={14}/> Admin Login</>}
            </button>
          </div>
        </header>

        {/* Prediction Input Block moved to top */}
        <div className="card">
          <h3 style={{ fontSize: '1.25rem', marginBottom: '1rem', color: 'var(--text-primary)' }}>Make Prediction</h3>
          <form onSubmit={handlePredict}>
            <div className="action-form">
              {inputFields.map((field) => (
                <div className="form-group" key={field.name}>
                  <label>{field.label}</label>
                  <input 
                    type="number" step={field.step} name={field.name} 
                    value={predictionData[field.name]} onChange={handleInputChange} 
                    className="input-field" 
                  />
                </div>
              ))}
              <div style={{ display: 'flex', alignItems: 'flex-end', paddingTop: '0.4rem' }}>
                <button type="submit" className="btn btn-filled" style={{ width: '100%', height: '42px', fontSize: '0.95rem' }}><Play size={18} /> Run Prediction</button>
              </div>
            </div>
          </form>
        </div>

        <div className="top-cards-grid">
          <CircularProgress
            label="Estimated Cost"
            value={predictionResult ? `$${predictionResult.estimated_cost.toLocaleString()}` : "—"}
            maxValue={50000}
            color="var(--primary-cyan)"
            icon={Calendar}
            subLabel="Based on current state"
          />
          <CircularProgress
            label="Expected Profit"
            value={predictionResult ? `$${predictionResult.expected_profit.toLocaleString()}` : "—"}
            maxValue={150000}
            color="var(--primary-blue)"
            icon={User}
            subLabel="Projected return"
          />
          <CircularProgress
            label="Recommended Action"
            value={predictionResult?.recommended_action || "Pending..."}
            maxValue={1}
            color="var(--primary-magenta)"
            icon={Activity}
            subLabel={predictionResult ? "Model Output" : "Run prediction"}
          />
        </div>

        {predictionResult?.q_values && (
          <div className="card">
            <h3 style={{ fontSize: '1.25rem', marginBottom: '1rem', color: 'var(--text-primary)' }}>Model Confidence Level</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '2rem' }}>
              {confidences.map((conf, idx) => (
                <div key={idx}>
                  <div className="flex-between" style={{ marginBottom: '0.5rem' }}>
                    <span style={{ fontSize: '0.95rem', color: 'var(--text-primary)' }}>{actionNames[idx]}</span>
                    <span style={{ fontSize: '0.95rem', fontWeight: 700, color: progressColors[idx] }}>{(conf * 100).toFixed(2)}%</span>
                  </div>
                  <div className="progress-bar-container">
                    <div className="progress-bar-fill" style={{ width: `${conf * 100}%`, backgroundColor: progressColors[idx] }}></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {showGraph && (
          <div className="card chart-section">
            <div className="flex-between">
              <div className="flex-row" style={{ gap: '2rem' }}>
                <div style={{ color: 'var(--danger)', fontSize: '0.9rem', fontWeight: 500 }}>■ Estimated Cost</div>
                <div style={{ color: 'var(--primary-blue)', fontSize: '0.9rem', fontWeight: 500 }}>■ Expected Profit</div>
              </div>
            </div>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={true} horizontal={false} stroke="var(--border-color)" />
                  <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} />
                  <Tooltip contentStyle={{ borderRadius: '8px', border: '1px solid var(--border-color)', backgroundColor: 'var(--card-bg)', color: 'var(--text-primary)' }} />
                  <Line type="monotone" dataKey="cost" stroke="var(--danger)" strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} />
                  <Line type="monotone" dataKey="profit" stroke="var(--primary-blue)" strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        <div className="card">
          <div className="flex-between" style={{ marginBottom: '0.75rem' }}>
            <h3 style={{ fontSize: '1.25rem', color: 'var(--text-primary)' }}>Activity Log</h3>
          </div>
          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr><th>Action / Event</th><th>Time</th><th>Status</th></tr>
              </thead>
              <tbody>
                {history.map((item) => (
                  <tr key={item.id}>
                    <td style={{ fontWeight: 500 }}>{item.action}</td>
                    <td>{item.date}</td>
                    <td>
                      <span style={{ color: item.status === 'Success' ? 'var(--success)' : 'var(--text-primary)' }}>{item.status}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </main>

      {isAdmin && (
        <aside className="admin-sidebar">
          <div className="flex-between" style={{ marginBottom: '1rem' }}>
            <h3 style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--primary-blue)' }}>Admin Controls</h3>
            <Lock size={20} color="var(--primary-blue)" />
          </div>
          
          <form onSubmit={handleUpload} className="sidebar-section">
            <label style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <FolderUp size={20} color="var(--primary-cyan)"/> Upload Dataset
            </label>
            <div className="file-upload-wrapper">
              <div className="file-upload-label">
                {file ? file.name : "Click to select CSV"}
              </div>
              <input type="file" onChange={e => setFile(e.target.files[0])} />
            </div>
            <button type="submit" className="btn btn-primary" style={{ width: '100%' }}><Upload size={18} /> Confirm Upload</button>
          </form>
          
          <div className="sidebar-section" style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1.5rem' }}>
            <label style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Cpu size={20} color="var(--primary-cyan)"/> Retrain Model
            </label>
            <button onClick={handleTrain} disabled={isTraining} className="btn btn-filled" style={{ width: '100%', padding: '1rem', fontSize: '1.1rem' }}>
              <Cpu size={18} /> {isTraining ? 'Training In Progress...' : 'Train'}
            </button>
          </div>

          {predictionResult?.q_values && (
            <div className="sidebar-section" style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1.5rem' }}>
              <label style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <BarChart3 size={20} color="var(--primary-cyan)"/> Q Values
              </label>
              <div className="q-value-row"><span>Do Nothing:</span> <span style={{ fontWeight: 600 }}>{predictionResult.q_values[0].toFixed(4)}</span></div>
              <div className="q-value-row"><span>Preventive:</span> <span style={{ fontWeight: 600 }}>{predictionResult.q_values[1].toFixed(4)}</span></div>
              <div className="q-value-row"><span>Corrective:</span> <span style={{ fontWeight: 600 }}>{predictionResult.q_values[2].toFixed(4)}</span></div>
            </div>
          )}
          <div className="sidebar-section" style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1.5rem' }}>
            <label style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
              <BarChart3 size={20} color="var(--primary-cyan)"/> Advanced Settings
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer', fontSize: '1rem', color: 'var(--text-primary)' }}>
              <input type="checkbox" checked={showGraph} onChange={(e) => setShowGraph(e.target.checked)} style={{ width: '18px', height: '18px', cursor: 'pointer' }} />
              Show Trend Graph
            </label>
          </div>
        </aside>
      )}

      {toast.show && (
        <div className={`toast ${toast.type}`}>
          {toast.message}
        </div>
      )}
    </div>
  );
}

export default App;
