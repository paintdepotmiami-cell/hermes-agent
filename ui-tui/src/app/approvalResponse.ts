import type { ApprovalReq } from '../types.js'

export function approvalRespondParams(req: ApprovalReq, choice: string, sessionId: string) {
  if (req.approvalId) {
    return {
      approval_id: req.approvalId,
      decision: choice === 'deny' ? 'deny' : 'approve_once'
    }
  }

  return { choice, session_id: sessionId }
}
