function helloWorld() {
    printl("Helllo world")
}

printl("Helllo world")
helloWorld()

//print(GetPosition().tostring())
//printl(GetPosition().tostring())

function PrintPlayerPosition(player) {
    local origin = player.GetOrigin(); // returns a Vector
    print("Player position: x = " + origin.x + ", y = " + origin.y + ", z = " + origin.z + "\n");
}

// Assuming this is being run per tick or on some event
function OnThink() {
    local player = Entities.FindByClassname(null, "player");
    if (player != null) {
        PrintPlayerPosition(player);
    }
}

function LogPlayerPosition() {
    local player = Entities.FindByClassname(null, "player");
    if (player != null) {
        local pos = player.GetOrigin();
        AddThinkToEnt(player, "PrintPlayerPosition")
        local msg = format("PlayerPos: x=%.2f y=%.2f z=%.2f", pos.x, pos.y, pos.z);
        printl(msg); // prints to console/log
    }
}
LogPlayerPosition()

//FileToString(string file);
StringToFile(string file, string string)
//SpawnEntity()
