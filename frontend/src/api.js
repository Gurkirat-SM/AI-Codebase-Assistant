import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

/** GET /api/index/status */
export const getIndexStatus = () => api.get('/index/status')

/** POST /api/index  { repo_path, persist } */
export const indexRepository = (repoPath, options = {}) =>
  api.post('/index', { repo_path: repoPath, persist: true, ...options })

/** POST /api/query  { query, top_k } */
export const queryRepository = (query, topK = 5, options = {}) =>
  api.post('/query', { query, top_k: topK, ...options })

