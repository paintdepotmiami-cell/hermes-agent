import { describe, expect, it } from 'vitest'

import { approvalRespondParams } from '../app/approvalResponse.js'

describe('approvalRespondParams', () => {
  it('sends only exact id and decision for a fresh approval', () => {
    expect(
      approvalRespondParams(
        {
          approvalId: 'fresh-id-17',
          command: 'sentinel operation',
          description: 'fresh approval'
        },
        'once',
        'runtime-session-client-value'
      )
    ).toEqual({ approval_id: 'fresh-id-17', decision: 'approve_once' })
  })

  it('keeps the established session payload for legacy approvals', () => {
    expect(
      approvalRespondParams(
        { command: 'sentinel operation', description: 'legacy approval' },
        'once',
        'runtime-session-1'
      )
    ).toEqual({ choice: 'once', session_id: 'runtime-session-1' })
  })
})
