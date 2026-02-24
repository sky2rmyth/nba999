-- Add game_date column to predictions table
ALTER TABLE predictions ADD COLUMN game_date DATE;

-- Backfill game_date from payload JSONB for existing rows
UPDATE predictions
SET game_date = (payload->>'game_date')::DATE
WHERE game_date IS NULL
  AND payload->>'game_date' IS NOT NULL;
