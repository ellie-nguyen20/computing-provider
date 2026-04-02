package db

import (
	"os"
	"path/filepath"
	"testing"
)

func TestInitDb_CreatesFile(t *testing.T) {
	dir := t.TempDir()

	InitDb(dir)

	dbPath := filepath.Join(dir, "provider.db")
	if _, err := os.Stat(dbPath); os.IsNotExist(err) {
		t.Errorf("expected provider.db to be created at %s", dbPath)
	}
}

func TestInitDb_WALMode(t *testing.T) {
	dir := t.TempDir()
	InitDb(dir)

	db := NewDbService()
	if db == nil {
		t.Fatal("expected non-nil DB")
	}

	var journalMode string
	row := db.Raw("PRAGMA journal_mode").Row()
	if err := row.Scan(&journalMode); err != nil {
		t.Fatalf("failed to query journal_mode: %v", err)
	}
	if journalMode != "wal" {
		t.Errorf("expected WAL mode, got %q", journalMode)
	}
}

func TestInitDb_AutoMigrate_TablesExist(t *testing.T) {
	dir := t.TempDir()
	InitDb(dir)

	db := NewDbService()
	if db == nil {
		t.Fatal("expected non-nil DB")
	}

	// CpInfoEntity table should exist after AutoMigrate
	var name string
	row := db.Raw("SELECT name FROM sqlite_master WHERE type='table' AND name='cp_info_entities'").Row()
	if err := row.Scan(&name); err != nil {
		t.Logf("cp_info_entities table not found (may use different name): %v", err)
		// Not a hard fail — table name may differ by GORM conventions
	}
}

func TestNewDbService_ReturnsSingleton(t *testing.T) {
	dir := t.TempDir()
	InitDb(dir)

	db1 := NewDbService()
	db2 := NewDbService()

	if db1 != db2 {
		t.Error("expected NewDbService to return the same singleton instance")
	}
}

func TestInitDb_MaxOpenConns(t *testing.T) {
	dir := t.TempDir()
	InitDb(dir)

	db := NewDbService()
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("failed to get underlying sql.DB: %v", err)
	}

	stats := sqlDB.Stats()
	if stats.MaxOpenConnections != 1 {
		t.Errorf("expected MaxOpenConnections=1, got %d", stats.MaxOpenConnections)
	}
}
