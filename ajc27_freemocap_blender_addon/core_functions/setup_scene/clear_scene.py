import traceback

def clear_scene_depr():
    ###%% clear the scene - Scorch the earth \o/
    import bpy

    print("Clearing scene...0")
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except:
        print("CLEAR 1 - \n\n\n\n\n\n\n\n")
        pass
    try:
        print("CLEAR 2 - \n\n\n\n\n\n\n\n")


        # bpy.ops.object.hide_view_clear()
        bpy.ops.object.select_all(action="SELECT")  # select all objects
        bpy.ops.object.delete(use_global=True)  # delete all objects from all scenes
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        # Clear data parent collection
        bpy.context.scene.freemocap_properties.data_parent_collection.clear()
        # Clear scene collections
        for collection in bpy.data.collections:
            print("removing collections")
            bpy.data.collections.remove(collection)
            
    except Exception as e:
        print("CLEAR 3- \n\n\n\n\n\n\n\n")
        traceback.print_exc()

        pass

def clear_scene_fixed():
    """Clear the scene safely - removes all objects and collections"""
    print("Clearing scene...")
    import bpy
    # 1. Switch to OBJECT mode if possible
    try:
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError as e:
        print(f"Warning: Could not switch to OBJECT mode: {e}")
    
    try:
        # 2. Deselect all first
        bpy.ops.object.select_all(action='DESELECT')
        
        # 3. Delete all objects from current scene
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False, confirm=False)
        
        # 4. Delete objects from ALL scenes (more reliable)
        for scene in bpy.data.scenes:
            for obj in list(scene.objects):  # Use list() to avoid modification during iteration
                try:
                    bpy.data.objects.remove(obj, do_unlink=True)
                except:
                    pass
        
        # 5. Purge orphan data
        for _ in range(3):  # Multiple passes for dependencies
            bpy.ops.outliner.orphans_purge(
                do_local_ids=True,
                do_linked_ids=True,
                do_recursive=True
            )
        
        # 6. Clear FreeMoCap properties if they exist
        if hasattr(bpy.context.scene, 'freemocap_properties'):
            if hasattr(bpy.context.scene.freemocap_properties, 'data_parent_collection'):
                bpy.context.scene.freemocap_properties.data_parent_collection.clear()
        
        # 7. Remove collections (use list() to avoid modification during iteration)
        for collection in list(bpy.data.collections):
            try:
                bpy.data.collections.remove(collection)
            except Exception as e:
                print(f"Warning: Could not remove collection {collection.name}: {e}")
        
        # 8. Verify context is still valid
        if not bpy.context.scene or not bpy.context.view_layer:
            print("ERROR: Context was invalidated during cleanup!")
            return False
        
        # print(f"✅ Scene cleared. Objects remaining: {len(bpy.context.scene.objects)}")
        return True
        
    except Exception as e:
        print(f"ERROR during scene cleanup: {e}")
        traceback.print_exc()
        return False


def clear_scene():
    """Completely reset the Blender context to a clean state"""
    clear_scene_fixed()



def nuclear_reset():
    """
    NUCLEAR OPTION: Delete absolutely everything and reset Blender to startup state
    WARNING: This will delete ALL objects, meshes, materials, textures, etc.
    """
    print("🔥 INITIATING NUCLEAR CLEANUP 🔥")
    import bpy
    # 1. Switch to OBJECT mode (required for deletion)
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # 2. Delete ALL objects in the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    
    # 3. Delete all orphaned data blocks (meshes, armatures, materials, etc.)
    for collection in [
        bpy.data.meshes,
        bpy.data.armatures,
        bpy.data.curves,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.materials,
        bpy.data.textures,
        bpy.data.images,
        bpy.data.actions,
        bpy.data.collections,
    ]:
        for item in collection:
            collection.remove(item)
    
    # 4. Purge orphan data (multiple passes to catch dependencies)
    for i in range(3):
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True,
            do_linked_ids=True,
            do_recursive=True
        )
    
    # 5. Clear active object
    bpy.context.view_layer.objects.active = None
    
    # 6. Update scene
    bpy.context.view_layer.update()
    
    print(f"✅ Cleanup complete. Objects remaining: {len(bpy.context.scene.objects)}")




def ultra_nuclear_reset():
    """
    ULTRA NUCLEAR: Reset to completely empty Blender file
    Deletes EVERYTHING including all scenes
    """
    print("💀 ULTRA NUCLEAR CLEANUP - DELETING EVERYTHING 💀")
    import bpy
    
    # Switch to object mode
    if bpy.context.object:
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Delete all objects from ALL scenes
    for scene in bpy.data.scenes:
        for obj in scene.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Delete all data blocks
    for bpy_data_collection in [
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.textures,
        bpy.data.images,
        bpy.data.brushes,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.armatures,
        bpy.data.curves,
        bpy.data.metaballs,
        bpy.data.lattices,
        bpy.data.fonts,
        bpy.data.grease_pencils,
        bpy.data.movieclips,
        bpy.data.sounds,
        bpy.data.speakers,
        bpy.data.lightprobes,
        bpy.data.collections,
        bpy.data.actions,
        bpy.data.particles,
    ]:
        for item in list(bpy_data_collection):
            bpy_data_collection.remove(item)
    
    # Multiple purge passes
    for i in range(5):
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True,
            do_linked_ids=True,
            do_recursive=True
        )
    
    # Reset to factory settings (optional - very aggressive)
    # bpy.ops.wm.read_factory_settings(use_empty=True)
    
    print(f"✅ Ultra cleanup complete. Objects: {len(bpy.data.objects)}")

def nuclear_reset_clean():
    """
    NUCLEAR OPTION: Delete everything but maintain valid Blender context
    """
    print("🔥 INITIATING NUCLEAR RESET 🔥")
    import bpy
    # 1. Switch to OBJECT mode (required for deletion)
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # 2. Deselect all and clear active
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = None
    
    # 3. Delete ALL objects in the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    
    # 4. Delete all orphaned data blocks
    for collection in [
        bpy.data.meshes,
        bpy.data.armatures,
        bpy.data.curves,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.materials,
        bpy.data.textures,
        bpy.data.images,
        bpy.data.actions,
    ]:
        for item in list(collection):  # Use list() to avoid modifying during iteration
            collection.remove(item)
    
    # 5. Purge orphan data (multiple passes)
    for i in range(3):
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True,
            do_linked_ids=True,
            do_recursive=True
        )
    
    # 6. CRITICAL: Ensure we have a valid scene and view layer
    if not bpy.context.scene:
        # Create a new scene if somehow deleted
        bpy.ops.scene.new(type='NEW')
    
    # 7. Ensure view layer exists
    if not bpy.context.view_layer:
        print("WARNING: No view layer, Blender state may be corrupted")
    
    # 8. Update scene
    bpy.context.view_layer.update()
    
    print(f"✅ Cleanup complete. Objects remaining: {len(bpy.context.scene.objects)}")